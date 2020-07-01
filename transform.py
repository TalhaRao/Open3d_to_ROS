#!/usr/bin/env python

'''

File name: transform.py
Author: Kanwar Talha Bin Liaqat
Email: kanwar.Liaqat@gmail.com
Date created: 18.06.2019
Date last modified:18.12.2019
Python Version: 2.7

'''

import rospy
import copy
import message_filters
import std_msgs
import geometry_msgs.msg
import tf

import lib.open3d       as op
import numpy            as np
import sensor_msgs.point_cloud2 as pc2

from sensor_msgs.msg import PointCloud2, PointField
from math import pi, atan2, asin, cos, sin


# ----------------------------------------------------------------------------------------
# Constants

HORIZONTAL_PIXELS 	= rospy.get_param('/ifx/icp/HORIZONTAL_PIXELS'	, 352)
VERTICAL_PIXELS 	= rospy.get_param('/ifx/icp/VERTICAL_PIXELS'	, 287)
HORIZONTAL_ANGLE 	= rospy.get_param('/ifx/icp/HORIZONTAL_ANGLE'	, 100)

RIGHT = 'right'
LEFT = 'left'

RAD_TO_DEG = 180.0/pi
DEG_TO_RAD = pi/180.0

trans_0 = np.asarray(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]])


pm1_trans = np.asarray(
    [[0.56994855, -0.00120133,  0.82168047,  0.05209416],
     [0.02044108,  0.99971085, -0.01271574,  0.01544554],
     [-0.82142628,  0.02404287,  0.56980766, -0.01864517],
     [0.0,         0.0,          0.0,          1.0]])

# 3rd ICP pm3
pm3_trans = np.asarray(
    [[0.5906445,   0.01829783, -0.8067244,  -0.08061128],
     [-0.00501105,  0.99980676,  0.01900842, -0.02117148],
     [0.80691634, -0.00718469,  0.59062206, -0.0225304],
     [0.0, 0.0, 0.0, 1.0]])

# ----------------------------------------------------------------------------------------
# Converts Transformation Matrix to tf_msg type
# INPUTS:
#       trans: Transformation Matrix
#       tof:   child frame id
#       faf:   header frame id

def matrix_to_tf_msg(trans, tof, faf):

    s = trans

    T = s[0:3, 3].ravel().tolist()
    # br = tf2.TransformBroadcaster() # Initializing the transform broadcaster
    header = std_msgs.msg.Header()  # assigning values for the header
    header.stamp = rospy.Time.now()
    header.frame_id = faf
    # msg = geometry_msgs.msg.TransformedStamped()
    msg = geometry_msgs.msg.TransformStamped()  # assiging values for the msg
    msg.header = header
    msg.child_frame_id = tof
    # translations
    msg.transform.translation.x = T[0]
    msg.transform.translation.y = T[1]
    msg.transform.translation.z = T[2]
    # rotations
    msg.transform.rotation.x = tf.transformations.quaternion_from_matrix(s)[0]
    msg.transform.rotation.y = tf.transformations.quaternion_from_matrix(s)[1]
    msg.transform.rotation.z = tf.transformations.quaternion_from_matrix(s)[2]
    msg.transform.rotation.w = tf.transformations.quaternion_from_matrix(s)[3]

    return msg


# ----------------------------------------------------------------------------------------
# Converts incoming transformation matrix tuple to tf_msg type

def tuple_to_tf(data, pm):

    if not isinstance(data, np.ndarray):

        tf_mat = np.reshape(data.trans_matrix, (-1, 4))

    else:
        tf_mat = data

    tf_msg = matrix_to_tf_msg(tf_mat, pm, 'ifx_pm2_optical_frame')
    return tf_msg


# ----------------------------------------------------------------------------------------
# Converts incoming transformation matrix tuple to 4x4 Matrix

def tuple_to_Matrix(data):

    if not isinstance(data, np.ndarray):

        tf_mat = np.reshape(data.trans_matrix, (-1, 4))

    else:
        tf_mat = data

    return tf_mat

# ----------------------------------------------------------------------------------------
# Function to draw 2 3D clouds with transformation

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    op.draw_geometries([source_temp, target_temp])


# ----------------------------------------------------------------------------------------
# Function to draw 3 3D clouds with transformation

def draw_registration_result3(pm1, pm2, pm3, transformation1, transformation3):
    pm1_temp = copy.deepcopy(pm1)
    pm2_temp = copy.deepcopy(pm2)
    pm3_temp = copy.deepcopy(pm3)

    pm1_temp.paint_uniform_color([1, 0.706, 0])
    pm2_temp.paint_uniform_color([0, 0.651, 0.929])
    pm3_temp.paint_uniform_color([1.5, 0.6, 0])

    pm1_temp.transform(transformation1)
    pm3_temp.transform(transformation3)
    op.draw_geometries([pm1_temp, pm2_temp, pm3_temp])


# # ----------------------------------------------------------------------------------------
# # Function to prepare source and target pointcloud for ICP

# def prepare_dataset(source, target, voxel_size):

#     source.transform(np.identity(4))
#     source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
#     target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
#     return source, target, source_down, target_down, source_fpfh, target_fpfh


# # ----------------------------------------------------------------------------------------
# # Function to downsample and find normals in the cloud

# def preprocess_point_cloud(pcd, voxel_size):

#     pcd_down = op.voxel_down_sample(pcd, voxel_size)
#     radius_normal = voxel_size * 2
#     # Estimate normal with search radius %.3f." % radius_normal
#     op.estimate_normals(pcd_down, op.KDTreeSearchParamHybrid(
#         radius=radius_normal, max_nn=30))

#     radius_feature = voxel_size * 5
#     # Compute FPFH feature with search radius %.3f." % radius_feature
#     pcd_fpfh = op.compute_fpfh_feature(pcd_down,
#                op.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
#     return pcd_down, pcd_fpfh


# ----------------------------------------------------------------------------------------
# Function perform fast global registration

def execute_fast_global_registration(source_down, target_down,
                                     source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    result = op.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        op.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


# ----------------------------------------------------------------------------------------
# Function to perform Point-To-Plane Registration

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, trans):
    distance_threshold = voxel_size * 0.3

    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = op.registration_icp(source, target, distance_threshold,
                                 trans, op.TransformationEstimationPointToPlane(),
                                 op.ICPConvergenceCriteria(max_iteration=1000))

    return result


# ----------------------------------------------------------------------------------------
# Function to perform Global Registration

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1

    print(":: RANSAC registration on downsampled point clouds.")
    result = op.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        distance_threshold,
        op.TransformationEstimationPointToPoint(), 4,
        [op.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         op.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        op.RANSACConvergenceCriteria(4000000, 500))

    return result

# ----------------------------------------------------------------------------------------
# Convert pcl Cloud to ROS PointCloud2


def ponits_to_ros(cam, points):

    ros_cloud = PointCloud2()

    ros_cloud.header = std_msgs.msg.Header()
    ros_cloud.header.stamp = rospy.Time.now()
    ros_cloud.header.frame_id = cam
    ros_cloud.height = 3
    ros_cloud.width = len(np.asarray(points, np.float32))
    ros_cloud.fields = [

        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)]

    ros_cloud.is_bigendian = False
    ros_cloud.point_step = 12

    ros_cloud.data = np.asarray(points, np.float32).tostring()

    return ros_cloud

# ----------------------------------------------------------------------------------------
# Convert ROS cloud into Points list
def ros_to_points_list(cloudIn):

    points_list = []

    for data in pc2.read_points(cloudIn, field_names=("x", "y", "z"), skip_nans=True):
        points_list.append([data[0], data[1], data[2]])

    return points_list

# ----------------------------------------------------------------------------------------
# Convert ROS cloud into Open3d cloud 

def ros_to_open3d(cloudIn):
    
    # Create Open3d.PointCloud object
    open3d_cloud    = op.PointCloud()
    points_list     = ros_to_points_list(cloudIn)

    full_cloud_array = np.asarray(points_list, np.float32)
    open3d_cloud.points = op.Vector3dVector(full_cloud_array)

    return open3d_cloud


# ----------------------------------------------------------------------------------------
# Function to downsample and find normals in the cloud

def preprocess_point_cloud(cloudIn, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = op.voxel_down_sample(cloudIn, voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    op.estimate_normals(pcd_down, op.KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = op.compute_fpfh_feature(pcd_down,
                                        op.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # open3d_cloud_filter = pcd_down
    # open3d_cloud_filter_normal = pcd_fpfh
    return pcd_down, pcd_fpfh
# ----------------------------------------------------------------------------------------
# Get filtered open3d point cloud
def get_open3d_filter_cloud(cloudIn, voxel_size):

    open3D_cloud = ros_to_open3d(cloudIn)
    open3d_cloud_filter, open3d_cloud_filter_normal = preprocess_point_cloud(open3D_cloud, voxel_size)

    return open3d_cloud_filter, open3d_cloud_filter_normal

# --------------------------------------------------------
# Create Transformation Matrix with roll pitch yaw and x,y,z values


def transformationMat(roll, pitch, yaw, x, y, z):

    R11 = cos(pitch)*cos(yaw)
    R12 = (sin(roll)*sin(pitch)*cos(yaw))-(cos(roll)*sin(yaw))
    R13 = (cos(roll)*sin(pitch)*cos(yaw))+(sin(roll)*sin(yaw))
    R21 = cos(pitch)*sin(yaw)
    R22 = (sin(roll)*sin(pitch)*sin(yaw))+(cos(roll)*cos(yaw))
    R23 = (cos(roll)*sin(pitch)*sin(yaw))-(sin(roll)*cos(yaw))
    R31 = -sin(pitch)
    R32 = sin(roll)*cos(pitch)
    R33 = cos(roll)*cos(pitch)

    T = np.asarray([[R11, R12, R13, x],
                    [R21, R22, R23, y],
                    [R31, R32, R33, z],
                    [0, 0, 0, 1]])

    return T

# --------------------------------------------------------
# Create Rotation Matrix with roll pitch yaw values


def rotationMat(roll, pitch, yaw):

    R11 = cos(pitch)*cos(yaw)
    R12 = (sin(roll)*sin(pitch)*cos(yaw))-(cos(roll)*sin(yaw))
    R13 = (cos(roll)*sin(pitch)*cos(yaw))+(sin(roll)*sin(yaw))
    R21 = cos(pitch)*sin(yaw)
    R22 = (sin(roll)*sin(pitch)*sin(yaw))+(cos(roll)*cos(yaw))
    R23 = (cos(roll)*sin(pitch)*sin(yaw))-(sin(roll)*cos(yaw))
    R31 = -sin(pitch)
    R32 = sin(roll)*cos(pitch)
    R33 = cos(roll)*cos(pitch)

    R = np.asarray([[R11, R12, R13],
                    [R21, R22, R23],
                    [R31, R32, R33]])

    return R


# --------------------------------------------------------
# Create Rotation Matrix with roll pitch yaw values


def robotMat(trans, glob):

    T11 = trans[0][0]-glob[0][0]+1
    T12 = trans[0][1]-glob[0][1]
    T13 = trans[0][2]-glob[0][2]
    T14 = trans[0][3]-glob[0][3]
    T21 = trans[1][0]-glob[1][0]
    T22 = trans[1][1]-glob[1][1]+1
    T23 = trans[1][2]-glob[1][2]
    T24 = trans[1][3]-glob[1][3]
    T31 = trans[2][0]-glob[2][0]
    T32 = trans[2][1]-glob[2][1]
    T33 = trans[2][2]-glob[2][2]+1
    T34 = trans[2][3]-glob[2][3]
    T41 = trans[3][0]-glob[3][0]
    T42 = trans[3][1]-glob[3][1]
    T43 = trans[3][2]-glob[3][2]
    T44 = trans[3][3]-glob[3][3]+1

    T = np.asarray([[T11, T12, T13, T14],
                    [T21, T22, T23, T24],
                    [T31, T32, T33, T34],
                    [T41, T42, T43, T44]])

    return T

# ----------------------------------------------------------------------------------------

# Custom class for cloud conversion


class cloud:

    def __init__(self):

        self.__points_list                  = []

        self.__open3d_cloud                 = op.PointCloud()
        self.__open3d_cloud_filter          = op.PointCloud()
        self.__open3d_cloud_filter_normal   = op.Feature()
        self.__opend3d_cropped_cloud        = op.PointCloud()
        self.__open3d_cropped_cloud_filter  = op.PointCloud()

        self.__ros_cloud                    = PointCloud2()

        # self.ros_to_open3d(cloudIn, direction, angle, cam)

    # ----------------------------------------------------------------------------------------

    def ros_to_points_list(self, cloudIn):

        points_list = []

        for data in pc2.read_points(cloudIn, field_names=("x", "y", "z"), skip_nans=True):
            points_list.append([data[0], data[1], data[2]])

        self.__points_list = points_list

    # ----------------------------------------------------------------------------------------

    def ros_to_open3d(self, cloudIn):
             # Create Open3d.PointCloud object

        self.ros_to_points_list(cloudIn)

        full_cloud_array = np.asarray(self.__points_list, np.float32)
        self.__open3d_cloud.points = op.Vector3dVector(full_cloud_array)

    # ----------------------------------------------------------------------------------------
    # Function to downsample and find normals in the cloud

    def preprocess_point_cloud(self, cloudIn, voxel_size):
            # print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = op.voxel_down_sample(cloudIn, voxel_size)

        radius_normal = voxel_size * 2
        # print(":: Estimate normal with search radius %.3f." % radius_normal)
        op.estimate_normals(pcd_down, op.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = op.compute_fpfh_feature(pcd_down,
                                           op.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        self.__open3d_cloud_filter = pcd_down
        self.__open3d_cloud_filter_normal = pcd_fpfh
        return pcd_down, pcd_fpfh

    # ----------------------------------------------------------------------------------------

    # def ros_to_pcl(self, cloudIn):

    #     points_list = []

    #     for data in pc2.read_points(cloudIn, field_names=("x", "y", "z"), skip_nans=True):
    #         points_list.append([data[0], data[1], data[2]])

    #     pcl_data = PointCloud()
    #     pcl_data.from_list(points_list)
    #     self.__pcl_cloud = pcl_data
    #     self.__points_list = points_list

    # ----------------------------------------------------------------------------------------
    # Convert Open3d Cloud to ROS PointCloud2

    def open3d_to_ros(self, cam):

        self.__ros_cloud.header = std_msgs.msg.Header()
        self.__ros_cloud.header.stamp = rospy.Time.now()
        self.__ros_cloud.header.frame_id = cam
        self.__ros_cloud.height = 1
        self.__ros_cloud.width = len(
            np.asarray(self.__points_list, np.float32))
        self.__ros_cloud.fields = [

            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]

        self.__ros_cloud.is_bigendian = False
        self.__ros_cloud.point_step = 12

        self.__ros_cloud.data = np.asarray(
            self.__points_list, np.float32).tostring()

        # print(len(self.__ros_cloud.data))

    # ----------------------------------------------------------------------------------------
    # Convert Open3d Cloud to ROS PointCloud2

    def cropped_ros_cloud(self, cloudIn, direction, angle, cam):

        cropped_cloud_array = self.crop_cloud(cloudIn, direction, angle)

        self.__ros_cloud.header = std_msgs.msg.Header()
        self.__ros_cloud.header.stamp = rospy.Time.now()
        self.__ros_cloud.header.frame_id = cam
        self.__ros_cloud.height = 1
        self.__ros_cloud.width = len(cropped_cloud_array)
        self.__ros_cloud.fields = [

            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]

        self.__ros_cloud.is_bigendian = False
        self.__ros_cloud.point_step = 12

        self.__ros_cloud.data = np.asarray(
            cropped_cloud_array, np.float32).tostring()

    # ----------------------------------------------------------------------------------------

    # def filter_cloud(self):

    #     approximate_voxel_filter = self.__pcl_cloud.make_ApproximateVoxelGrid()
    #     approximate_voxel_filter_cropped = self.__pcl_cropped_cloud.make_ApproximateVoxelGrid()
    #     approximate_voxel_filter.set_leaf_size(0.05, 0.05, 0.05)
    #     approximate_voxel_filter_cropped.set_leaf_size(0.05, 0.05, 0.05)
    #     self.__pcl_cloud_filter = approximate_voxel_filter.filter()
    #     self.__pcl_cropped_cloud_filter = approximate_voxel_filter_cropped.filter()

        # print('Filtered cloud contains ' + str(self.__pcl_cloud_filter.size) + ' data points from room_scan2.pcd')

    # ----------------------------------------------------------------------------------------

    def crop_cloud(self, cloudIn, direction, angle):

        self.ros_to_points_list(cloudIn)

        full_cloud_array = np.asarray(self.__points_list, np.float32)

        print(self.__points_list.__sizeof__)

        x_full = np.reshape(full_cloud_array[:, 0], (-1, HORIZONTAL_PIXELS))
        y_full = np.reshape(full_cloud_array[:, 1], (-1, HORIZONTAL_PIXELS))
        z_full = np.reshape(full_cloud_array[:, 2], (-1, HORIZONTAL_PIXELS))

        # Convert to 3D-array of dimention [171x224x3]
        cloud_3d_array = np.dstack([x_full, y_full, z_full])

        # Based on the angle given, calculate the no. of columns to cut
        no_of_columns = int(
            (HORIZONTAL_PIXELS/float(HORIZONTAL_ANGLE)) * angle)

        if(direction == LEFT):

            cropped_3d_array = cloud_3d_array[:, : no_of_columns]

        else:
            cropped_3d_array = cloud_3d_array[:,
                                              HORIZONTAL_PIXELS - no_of_columns:]

        x, y, z = np.moveaxis(cropped_3d_array, -1, 0)
        x_cropped = np.reshape(x, (-1, (VERTICAL_PIXELS * no_of_columns)))
        y_cropped = np.reshape(y, (-1, (VERTICAL_PIXELS * no_of_columns)))
        z_cropped = np.reshape(z, (-1, (VERTICAL_PIXELS * no_of_columns)))

        # Dimention = [cropped_columns x 3]
        cropped_cloud_array = np.vstack((x_cropped, y_cropped, z_cropped)).T

        return cropped_cloud_array
        # self.__ros_cropped_cloud = cropped_cloud_array
        # self.__opend3d_cropped_cloud.points = op.Vector3dVector(cropped_cloud_array)

    # ----------------------------------------------------------------------------------------

    def get_open3d_cloud(self, cloudIn):

        self.ros_to_open3d(cloudIn)
        return self.__open3d_cloud

    # ----------------------------------------------------------------------------------------

    def get_open3d_filter_cloud(self, cloudIn, voxel_size):

        open3D_cloud = self.get_open3d_cloud(cloudIn)
        self.preprocess_point_cloud(open3D_cloud, voxel_size)

        return self.__open3d_cloud_filter, self.__open3d_cloud_filter_normal

    # ----------------------------------------------------------------------------------------

    def get_cloud_normals(self, cloudIn, voxel_size):

        if(self.__open3d_cloud_filter_normal.num() == 0 and self.__open3d_cloud_filter_normal.dimension() == 0):
            self.get_open3d_filter_cloud(cloudIn, voxel_size)

        return self.__open3d_cloud_filter_normal

    # ----------------------------------------------------------------------------------------

    # def get_pcl_cloud(self):

    #     return self.__pcl_cloud

    # ----------------------------------------------------------------------------------------

    def get_points_list(self):

        return self.__points_list

    # ----------------------------------------------------------------------------------------

    # def get_pcl_filter_cloud(self):

    #     return self.__pcl_cloud_filter

    # ----------------------------------------------------------------------------------------

    def get_open3d_cropped_cloud(self):

        return self.__opend3d_cropped_cloud

    # ----------------------------------------------------------------------------------------

    def get_open3d_cropped_filter_cloud(self):

        return self.__open3d_cropped_cloud_filter

    # ----------------------------------------------------------------------------------------

    def get_ros_cloud(self, cloudIn, direction, angle, cam):

        self.cropped_ros_cloud(cloudIn, direction, angle, cam)
        return self.__ros_cloud
