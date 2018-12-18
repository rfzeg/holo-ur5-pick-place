#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
from math import *
from time import sleep
import sys
import copy
import moveit_msgs.msg
import geometry_msgs.msg
import roslib
from std_msgs.msg import String, Int32MultiArray
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from moveit_msgs.msg import Grasp, GripperTranslation, PlaceLocation
from control_msgs.msg import *
from trajectory_msgs.msg import *
from sensor_msgs.msg import JointState
import actionlib
import rospy
from std_msgs.msg import String, Header
from geometry_msgs.msg import WrenchStamped, Vector3
import tf
from tf.transformations import *
from math import pi
from inverseKinematicsUR5 import InverseKinematicsUR5, transformRobotParameter
from copy import deepcopy
from std_msgs.msg import Int32
from ropi_msgs.srv import GripperControl

m_cali = np.array([
    [1, 0, 0, 0.715],
    [0, 1, 0, -0.311],
    [0, 0, 1, 0.188],
    [0, 0, 0, 1.0]
])
JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
SPEED = 3

def left2right(mat):
    s = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    rot_ = mat[:3, :3]
    trans = mat[:3, 3]
    rot_ = np.matmul(np.matmul(s, rot_), s)
    trans_ = np.matmul(s, trans)
    mat_ = np.zeros((4,4))
    mat_[:3, :3] = rot_
    mat_[:3, 3] = trans_
    mat_[3, 3] = 1
    return mat_ 

def transform_inverse(mat):
    rot = mat[:3, :3]
    trans = mat[:3, -1]
    inv = np.identity(4)
    inv[:3, :3] = rot.T
    inv[:3, -1] = -np.matmul(rot.T, trans.reshape(-1, 1)).ravel()
    return inv

def compute_transform(m_cali, m, m_tag):
    m_tag_inv = tf.transformations.inverse_matrix(m_tag)
    temp = np.matmul(m_cali, m_tag_inv)
    return np.matmul(temp, m)

class GripperServiceClient(object):
    def __init__(self, topic):
        rospy.wait_for_service(topic)
        self.srv_proxy = rospy.ServiceProxy(topic, GripperControl)

    def move_to(self, pos):
        pos = 255 - int(pos / 0.14 * 255)
        try:
            resp1 = self.srv_proxy(pos)
            return resp1.success
        except rospy.ServiceException, e:
            print ("Service call failed: %s" % e)

    def close(self):
        try:
            resp1 = self.srv_proxy(255)
            return resp1.success
        except rospy.ServiceException, e:
            print ("Service call failed: %s" % e)

    def open(self):
        try:
            resp1 = self.srv_proxy(0)
            return resp1.success
        except rospy.ServiceException, e:
            print ("Service call failed: %s" % e)

#tool0 is ee
class pick_place:
    def __init__(self):
        #/vel_based_pos_traj_controller/
        #==========================init UR5==================================
        self.client = actionlib.SimpleActionClient('icl_phri_ur5/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.goal = FollowJointTrajectoryGoal()
        self.goal.trajectory = JointTrajectory()
        self.goal.trajectory.joint_names = JOINT_NAMES
        print ("Waiting for server...")
        self.client.wait_for_server()
        print ("Connected to server")
        joint_states = rospy.wait_for_message("joint_states", JointState)
        print(joint_states)
        joint_states = list(deepcopy(joint_states).position)
        del joint_states[-1]
        self.joints_pos_start = np.array(joint_states)
        print ("Init done")
        #==========================init tf==================================
        self.listener = tf.TransformListener()
        self.Transformer = tf.TransformerROS()
        self.broadcaster = tf.TransformBroadcaster()
	self.base_link = '/base_link'
	self.ee_link = '/ee_link'
        #==========================init IK==================================
        joint_weights = [12,5,4,3,2,1]
        self.ik = InverseKinematicsUR5()
        self.ik.setJointWeights(joint_weights)
        self.ik.setJointLimits(-pi, pi)
        #==========================init subscriber==================================
        self.sub_pickplace = rospy.Subscriber('/CHOICE', String, self.pickplace_cb)
        #==========================init Gripper==================================
        rospy.loginfo('Waiting for gripper service.')
        self.gripper_sc = GripperServiceClient('/gripper_control')
        self.gripper_sc.open()
        rospy.loginfo('finish init')

    def converter(self, position, euler):
        quat = tf.transformations.quaternion_from_euler(euler[0], euler[1], euler[2])
        mat = self.Transformer.fromTranslationRotation(position, quat)
        return mat
    
    def broad(self, mat, name):
        new_mat = np.zeros((4,4))
        new_mat[:3, :3] = mat[:3, :3]
        new_mat[3 ,3] = 1
        quat = tf.transformations.quaternion_from_matrix(new_mat)
        self.broadcaster.sendTransform(mat[:3, -1],
                         quat,
                         rospy.Time.now(),
                         name,
                         "/base_link")

    # euler from tool0 to base_link: 3.14, 0, alpha
    def output_grasp(self, mat):
        quat = tf.transformations.quaternion_from_euler(3.14, 0, -3.14)
        pos = mat[:3, 3]
        pos[-1] = 0.188
        mat = self.Transformer.fromTranslationRotation(pos, quat)
        return mat

    # pos,pos,pos|euler|pos,pos,pos|euler|pos,pos,pos 
    def string_parser(self, strings):
        temp = strings.split("|")
        pick_pos = map(float, temp[0].split(","))
        place_pos = map(float, temp[2].split(","))
        tag_pos = map(float, temp[4].split(","))

        pick_euler = map(float, temp[1][1:-1].split(","))
        place_euler = map(float, temp[3][1:-1].split(","))
        tag_euler = (0, 0, 0)
        return [self.converter(pick_pos, pick_euler), self.converter(place_pos, place_euler), self.converter(tag_pos, tag_euler)]
    
    def move(self, dest_m):

        qsol = self.ik.findClosestIK(dest_m,self.joints_pos_start)
        
        if qsol is not None:
            if qsol[0] < 0:
                qsol[0] += pi
            else:
                qsol[0] -= pi
            self.goal.trajectory.points = [
                JointTrajectoryPoint(positions=self.joints_pos_start.tolist(), velocities=[0]*6, time_from_start=rospy.Duration(0.0)),
                JointTrajectoryPoint(positions=qsol.tolist(), velocities=[0]*6, time_from_start=rospy.Duration(SPEED)),
            ]

            try:
                self.client.send_goal(self.goal)
                self.joints_pos_start = qsol
                self.client.wait_for_result()
            except:
                raise
        elif qsol is None:
            rospy.loginfo("fail to find IK solution")

    def pickplace_cb(self, msg):
        if msg:
            strings = msg.data
            m_pick, m_place, m_tag = self.string_parser(strings)
            m_pick = left2right(m_pick)
            m_place = left2right(m_place)
            m_tag = left2right(m_tag)
            pick = compute_transform(m_cali, m_pick, m_tag)
	    position, quaternion = self.listener.lookupTransform(self.base_link, self.ee_link,rospy.Time(0))
	    m_br = self.Transformer.fromTranslationRotation(position, quat)
            pick = np.matmul(m_br,pick)
            # move to pick here
            place = np.matmul(pick,m_place)
            # move to place here

if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)
    task = pick_place()
    rospy.spin()
