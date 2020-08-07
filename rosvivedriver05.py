"""
This is VoltBro OpenVR Driver for ROS
author: Roman Scherbov
romanbt405@gmail.com
Thanx for ideas and initial code:
https://github.com/TriadSemi/triad_openvr 
https://gist.github.com/awesomebytes/75daab3adb62b331f21ecf3a03b3ab46
"""

import sys
import time
import openvr
import numpy
import math

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
import tf
from tf.transformations import quaternion_from_euler, quaternion_multiply, euler_from_quaternion


class DriverOfVRsystem():

    def __init__(self):

        self.left_controller_id_in_VRsystem = None
        self.right_controller_id_in_VRsystem = None
        max_init_retries = 4
        retries = 0
        print("===========================")
        while retries < max_init_retries:
            try:
                print("Initializing OpenVR...")
                openvr.init(openvr.VRApplication_Scene)
                break
            except openvr.OpenVRError as e:
                print("Error while initializing OpenVR (try {} / {})".format(
                    retries + 1, max_init_retries))
                print(e)
                retries += 1
                time.sleep(4.0)
            else:
                print("Could not initialize OpenVR, aborting.")
                exit(0)
        
        print("Success!")
        print("===========================")
        self.vrsystem = openvr.VRSystem()
        self.show_only_new_events = True
        self.last_unPacketNum_left = 0
        self.last_unPacketNum_right = 0
        self.controller_inititalization()

    def get_controllers_id(self):
        """
        We need to find out which indexes do both controllers have inside VRSystem. 
        So we are asking openvr by index for role of each device found inside the initialized VRsystem
        and if their roles identified as controller save such indexes as controllers indexes.
        NB! Controllers should be in LOS with base satations for initialization.
        """

        for i in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = self.vrsystem.getTrackedDeviceClass(i)
            if device_class == openvr.TrackedDeviceClass_Controller:
                role = self.vrsystem.getControllerRoleForTrackedDeviceIndex(i)
                if role == openvr.TrackedControllerRole_RightHand:
                    self.right_controller_id_in_VRsystem = i
                if role == openvr.TrackedControllerRole_LeftHand:
                    self.left_controller_id_in_VRsystem = i

    def controller_inititalization(self):
        """
        Waiting for BOTH controllers to get visible so we can read their data
        """

        print("===========================")
        print("Waiting for BOTH controllers to get visible...")
        try:
            while self.left_controller_id_in_VRsystem is None or self.right_controller_id_in_VRsystem is None:
                self.get_controllers_id()
                if self.left_controller_id_in_VRsystem and self.right_controller_id_in_VRsystem:
                    print("Controllers initialized")
                    break
                print("Waiting for BOTH controllers to get visible...")
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Control+C pressed, shutting down...")
            openvr.shutdown()
        return [self.left_controller_id_in_VRsystem, self.right_controller_id_in_VRsystem]

    def convert_controller_state_to_dict(self, pControllerState):
        """
        Converting message got from vrsystem.getControllerState()
        docs: https://github.com/ValveSoftware/openvr/wiki/IVRSystem::GetControllerState
        """

        d = {}
        d['unPacketNum'] = pControllerState.unPacketNum
        # on trigger .y is always 0.0 says the docs
        d['trigger'] = pControllerState.rAxis[1].x
        # 0.0 on trigger is fully released
        # -1.0 to 1.0 on joystick and trackpads
        d['trackpad_x'] = pControllerState.rAxis[0].x
        d['trackpad_y'] = pControllerState.rAxis[0].y
        # These are published and always 0.0
        # for i in range(2, 5):
        #     d['unknowns_' + str(i) + '_x'] = pControllerState.rAxis[i].x
        #     d['unknowns_' + str(i) + '_y'] = pControllerState.rAxis[i].y
        d['ulButtonPressed'] = pControllerState.ulButtonPressed
        d['ulButtonTouched'] = pControllerState.ulButtonTouched
        # To make easier to understand what is going on
        # Second bit marks menu button
        d['menu_button'] = bool(pControllerState.ulButtonPressed >> 1 & 1)
        # 32 bit marks trackpad
        d['trackpad_pressed'] = bool(pControllerState.ulButtonPressed >> 32 & 1)
        d['trackpad_touched'] = bool(pControllerState.ulButtonTouched >> 32 & 1)
        # third bit marks grip button
        d['grip_button'] = bool(pControllerState.ulButtonPressed >> 2 & 1)
        # System button can't be read, if you press it
        # the controllers stop reporting
        return d
    
    def button_press_handling(self, controller_id_in_VRsystem):

        show_only_new_events = True

        result, pControllerState = self.vrsystem.getControllerState(controller_id_in_VRsystem)
        d = self.convert_controller_state_to_dict(pControllerState)
        if show_only_new_events and self.last_unPacketNum_left != d['unPacketNum']:
            self.last_unPacketNum_left = d['unPacketNum']
            return d

    def get_devices_poses_and_extract_matrix_from_them(self):
        """
        Getting poses from VR system. Poses as [list] contain all 
        geo-data of devices which is registred in VRSystem. 
        To extract data for exact device we need to get it from poses[] 
        with device's index obtained from get_controllers_id(). 
        """

        poses = []
        poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)
        hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
        right_controller_pose = poses[self.right_controller_id_in_VRsystem]
        left_controller_pose = poses[self.left_controller_id_in_VRsystem]
        self.hmd_matrix = hmd_pose.mDeviceToAbsoluteTracking
        self.right_cont_matrix = right_controller_pose.mDeviceToAbsoluteTracking
        self.left_cont_matrix = left_controller_pose.mDeviceToAbsoluteTracking

    def convert_pose_matrix_to_euler_angles(self, pose_matrix):
        """
        Return Euler angles in degrees
        """

        R = [[0 for x in range(3)] for y in range(3)]
        for i in range(0,3):
            for j in range(0,3):
                R[i][j] = pose_matrix[i][j]
        
        R = numpy.asarray(R)

        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return numpy.array([math.degrees(x), math.degrees(y), math.degrees(z)])
    
    def convert_matrix_to_xyz_and_quaternion(self, pose_mat):
        """
        Return data for Quaternion() and Point() msg types
        """

        r_w = math.sqrt(abs(1+pose_mat[0][0]+pose_mat[1][1]+pose_mat[2][2]))/2
        r_x = (pose_mat[2][1]-pose_mat[1][2])/(4*r_w)
        r_y = (pose_mat[0][2]-pose_mat[2][0])/(4*r_w)
        r_z = (pose_mat[1][0]-pose_mat[0][1])/(4*r_w)

        x = pose_mat[0][3]
        y = pose_mat[1][3]
        z = pose_mat[2][3]
        return [x,y,z,r_x,r_y,r_z,r_w]

class ROSConnector():

    def __init__(self):
        """
        ROS inits basically
        """

        rospy.init_node("vive_node")
        self.rcont_pub = rospy.Publisher("/controller_pose", Pose, queue_size=1)
        self.hmd_pub = rospy.Publisher("hmd_pose", Pose, queue_size=1)
        self.cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
    
    def assembling_Pose_for_device(self, device_pose_matrix):
        """
        For now Pose is IK plugin end-effector's goal msg type. So we are assembling it for future use.
        """
        pose_to_post_in_ros_topic = Pose()
        #pose_to_post_in_ros_topic.header.frame_id = "base_scan"
        [cx, cy, cz, cqx, cqy, cqz, cqw] = drv.convert_matrix_to_xyz_and_quaternion(device_pose_matrix)
        pose_to_post_in_ros_topic = Pose(Point(cx, cy, cz), Quaternion(cqx,cqy,cqz,cqw))
        return pose_to_post_in_ros_topic
    
    def assembling_Twist_from_pose(self , device_pose_matrix):
        """
        Assembling Twist() type for publishing it in main loop
        """

        Kx = 0.2 # coefficient for linear.x of Left controller trigger button pressing
        Kz = 1.8 # coefficient for angular.z
        cmd_to_post_into_cmd_vel = Twist()
        # sending left controller id to openVR for reading buttons
        dict_of_buttons_pressed = drv.button_press_handling(drv.left_controller_id_in_VRsystem)
        # Menu button press escaping
        if dict_of_buttons_pressed != None: #
            key_motion_allower = dict_of_buttons_pressed['trackpad_touched']
            trigger_x_axis = dict_of_buttons_pressed['trigger']
            #Z - angle component of controller's pose
            Z = drv.convert_pose_matrix_to_euler_angles(device_pose_matrix)[2] 
            #if operator is not touching pad no movement allowed
            if key_motion_allower:
                cmd_to_post_into_cmd_vel.linear.x = trigger_x_axis * Kx
                # setting zero-zone for controller
                if -15 < Z and Z < 15:
                    cmd_to_post_into_cmd_vel.angular.z = 0
                else:
                    cmd_to_post_into_cmd_vel.angular.z = Z/180 * Kz
        else:
            cmd_to_post_into_cmd_vel.angular.z = 0
            cmd_to_post_into_cmd_vel.linear.x = 0

        return cmd_to_post_into_cmd_vel
    
    def main_ROS_loop(self):
        """
        Main "loop" and controller
        """

        while not rospy.is_shutdown():
            #getting all devices poses
            drv.get_devices_poses_and_extract_matrix_from_them()
            
            #pose matrix for HMD
            pose_of_hmd_to_pub = self.assembling_Pose_for_device(drv.hmd_matrix)

            #pose matrix for left controller
            twist_of_left_controller_to_pub = self.assembling_Twist_from_pose(drv.left_cont_matrix)

            #pose matrix for right controller
            pose_of_right_controller_to_pub = self.assembling_Pose_for_device(drv.right_cont_matrix)

            self.cmd_pub.publish(twist_of_left_controller_to_pub)
            self.hmd_pub.publish(pose_of_hmd_to_pub)
            self.rcont_pub.publish(pose_of_right_controller_to_pub)
            rospy.sleep(0.1)
        rospy.spin()

drv = DriverOfVRsystem()
r = ROSConnector()
r.main_ROS_loop()
openvr.shutdown()