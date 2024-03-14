import os
import sys
import shutil
import pandas as pd
import rosbag
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# ros
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.append(ros_path)

import cv2
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import *
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

if ros_path in sys.path:
    sys.path.remove(ros_path)

from dataset_utils import *
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

blue = lambda x: '\033[94m' + x + '\033[0m'
green = lambda x: '\033[1;32m' + x + '\033[0m'
yellow = lambda x: '\033[1;33m' + x + '\033[0m'
red = lambda x: '\033[1;31m' + x + '\033[0m'


class RosVisualizer():
    """
    1. 初始化ros节点，定义发布的话题类型，定义数据集路径
    2. 根据交互得到的scene的id，和类别，然后遍历所有，发布每一帧的点云以及box框信息，以及box中心点信息，均以ros话题形式发布

    注意：
        1. 数据集的路径dataset_path下应该类似以下目录格式
            .
            ├── calib
            ├── image_02
            ├── label_02
            └── velodyne

        2. 默认的话题发布频率需要手动设置，默认为10HZ
    """

    def __init__(self, args):
        # Init ros
        rospy.init_node('Ros_Visualizer', anonymous=True)
        self.box_marker_pub = rospy.Publisher('/predict_bbox', MarkerArray, queue_size=1)
        self.lidar_pub = rospy.Publisher('/bbox_lidar_3d', BoundingBoxArray, queue_size=10)
        self.lidar2d_pub = rospy.Publisher('/bbox_lidar', BoundingBoxArray, queue_size=10)

        self.point_pub = rospy.Publisher('/kitti/velo/pointcloud', PointCloud2, queue_size=1)
        self.box_center_pub = rospy.Publisher('/box_centers', String, queue_size=1)

        self._bboxpub = rospy.Publisher('/bbox_image_gt', BoundingBoxArray, queue_size=2)

        self.rate = rospy.Rate(args.pubRate)
        self.imagepub = rospy.Publisher('/kitti/camera_color_left/image_raw', Image, queue_size=2)
        self.projected_imagepub = rospy.Publisher('/projected_image', Image, queue_size=2)
        self._bridge = CvBridge()
        self.count = 1
        self.cv_image = None

        # 检查文件是否存在以确定是否是第一次写入
        self.bag_filename = os.path.join('/home/sam/', 'kitti003d.bag')
        self.is_first_write = not os.path.exists(self.bag_filename)

        # Init Attributes
        self.KITTI_Folder = args.dataset_path
        self.KITTI_img = os.path.join(self.KITTI_Folder, "image_02")
        self.KITTI_velo = os.path.join(self.KITTI_Folder, "velodyne")
        self.KITTI_label = os.path.join(self.KITTI_Folder, "label_02")
        self.replace = args.replace
        self.category = args.category
        self.save_path = args.save_path
        self.save_pcd = args.save_pcd

        print("KITTI_velo_path: ", self.KITTI_velo)
        print("KITTI_label_path: ", self.KITTI_label)

        self.width = 1242
        self.height = 375
        # 相机内参矩阵 K
        self.K = np.array([
            [9.597910e+02, 0.000000e+00, 6.960217e+02],
            [0.000000e+00, 9.569251e+02, 2.241806e+02],
            [0.000000e+00, 0.000000e+00, 1.000000e+00]
        ])

        # 相机畸变系数 D
        self.D = np.array([-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02])

        # LiDAR 到相机 00 的外参
        R1 = np.array([
            [7.533745e-03, -9.999714e-01, -6.166020e-04],
            [1.480249e-02, 7.280733e-04, -9.998902e-01],
            [9.998621e-01, 7.523790e-03, 1.480755e-02]
        ])
        T1 = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])

        # 相机 00 到相机 02 的外参
        R2 = np.array([
            [9.999758e-01, -5.267463e-03, -4.552439e-03],
            [5.251945e-03, 9.999804e-01, -3.413835e-03],
            [4.570332e-03, 3.389843e-03, 9.999838e-01]
        ])
        T2 = np.array([5.956621e-02, 2.900141e-04, 2.577209e-03])

        # 计算变换矩阵
        T_final = np.dot(R2, np.hstack((R1, T1.reshape(-1, 1)))) + T2.reshape(-1, 1)
        
        # 提取旋转矩阵 R_final (前三行和前三列)
        self.R = T_final[:3, :3]

        # 提取平移向量 T_final (第四列的前三行)
        self.T = T_final[:3, 3]


        # 相机内参矩阵 (P_rect_02)
        self.P_rect_02 = np.array([
            [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
            [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
            [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
        ])

        # 矫正旋转矩阵 (R_rect_00)
        self.R_rect_00 = np.array([
            [9.999239e-01, 9.837760e-03, -7.445048e-03],
            [-9.869795e-03, 9.999421e-01, -4.278459e-03],
            [7.402527e-03, 4.351614e-03, 9.999631e-01]
        ])
        self.R_rect_00 = np.vstack((self.R_rect_00, [0, 0, 0]))
        self.R_rect_00 = np.hstack((self.R_rect_00, np.array([[0], [0], [0], [1]])))

        # LiDAR 到相机的变换矩阵 (Tr_velo_to_cam)
        R_velo_to_cam = np.array([
            [7.533745e-03, -9.999714e-01, -6.166020e-04],
            [1.480249e-02, 7.280733e-04, -9.998902e-01],
            [9.998621e-01, 7.523790e-03, 1.480755e-02]
        ])
        T_velo_to_cam = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])

        self.Tr_velo_to_cam = np.hstack((R_velo_to_cam, T_velo_to_cam.reshape(-1, 1)))  # 3x4 矩阵
        self.Tr_velo_to_cam = np.vstack((self.Tr_velo_to_cam, [0, 0, 0, 1]))  # 使其成为 4x4 齐次变换矩阵

    def bbox3d_to_8points(self, bbox3d):
        # bbox3d中心点
        center = np.array([bbox3d.pose.position.x, bbox3d.pose.position.y, bbox3d.pose.position.z])
        # bbox3d尺寸
        size = np.array([bbox3d.dimensions.x, bbox3d.dimensions.y, bbox3d.dimensions.z])
        
        # 计算8个顶点
        dx = size[0] / 2
        dy = size[1] / 2
        dz = size[2] / 2
        corners = np.array([
            [dx, dy, dz], [dx, dy, -dz], [dx, -dy, dz], [dx, -dy, -dz],
            [-dx, dy, dz], [-dx, dy, -dz], [-dx, -dy, dz], [-dx, -dy, -dz]
        ])
        
        # 将bbox3d的中心点加到每个角点上
        corners = corners + center
        
        return corners
    
    def transform_and_project_bbox3d(self, bbox3d):
        corners = self.bbox3d_to_8points(bbox3d)
        
        # 将 corners 的每个点转换为齐次坐标
        ones = np.ones((corners.shape[0], 1))
        corners_homogeneous = np.hstack((corners, ones))
        
        # 应用变换
        points_camera_homogeneous = np.dot(self.P_rect_02, np.dot(self.R_rect_00, np.dot(self.Tr_velo_to_cam, corners_homogeneous.T)))
        
        # 保留位于相机前方的点
        front_points = points_camera_homogeneous[:, points_camera_homogeneous[2, :] > 0]
        
        # 检查是否还有剩余的点
        if front_points.shape[1] == 0:
            return None  # 如果没有，则忽略这个边界框
        
        # 进行归一化以得到2D像素坐标
        front_points = front_points[:2, :] / front_points[2, :]
        
        # 找到 2D 边界框
        # height, width = cv_image.shape[:2]
        xmin = max(0, np.min(front_points[0, :]))
        ymin = max(0, np.min(front_points[1, :]))
        xmax = min(self.width - 1, np.max(front_points[0, :]))
        ymax = min(self.height - 1, np.max(front_points[1, :]))
        
        if xmin >= xmax or ymin >= ymax:
            return None
        
        # 将边界框坐标转换为整数
        bbox_2d = np.array([xmin, ymin, xmax, ymax], dtype=int)
        
        return bbox_2d
    
    def get_sceneID(self):
        try:
            # input_type = str(input("please input the scene split type[number/dataset]\n"))
            input_type = 'number'
            if input_type.upper() == 'number'.upper():
                print("valid scenes are: \n", os.listdir(self.KITTI_velo))
                scene = int(input("please input the scene number\n"))
                sceneID = scene
            elif input_type.upper() == 'dataset'.upper():
                scene = input("please input the dataset type['train'/'test'/'valid'/'all']\n")
                sceneID = self.getSceneList(scene)
            else:
                sceneID = None
                print(red("Input Error!!\n"), "please run again and input 'number' or 'dataset'\n")
                exit()
            print(yellow("sceneID is:"), sceneID)
            return sceneID
        except:
            print("something error! Exiting...")
            exit()
    def pub_image(self, scene_id):
        image_path = os.path.join(self.KITTI_Folder, 'image_02')
        self.make_sure_path_valid(image_path)
        image_filenames = sorted(os.listdir(image_path))
        iterable = image_filenames

        scene_id = [scene_id] if isinstance(scene_id, int) else scene_id
        list_of_scene = [
            path for path in os.listdir(self.KITTI_velo)
            if os.path.isdir(os.path.join(self.KITTI_velo, path)) and
               int(path) in scene_id
        ]

        for filename in iterable:
            image_filename = os.path.join(image_path, filename)
            cv_image = cv2.imread(image_filename)

    def pub_pc_and_box(self, scene_id):
        pcd_path = os.path.join(self.save_path, self.category, 'lidar')
        label_path = os.path.join(self.save_path, self.category, 'label')
        image_path = os.path.join(self.KITTI_Folder, 'image_02')
        
        self.make_sure_path_valid(image_path)
        self.make_sure_path_valid(pcd_path)
        self.make_sure_path_valid(label_path)

        if self.save_pcd and self.replace is True:
            shutil.rmtree(pcd_path)
            shutil.rmtree(label_path)
            os.mkdir(pcd_path)
            os.mkdir(label_path)

        scene_id = [scene_id] if isinstance(scene_id, int) else scene_id
        list_of_scene = [
            path for path in os.listdir(self.KITTI_velo)
            if os.path.isdir(os.path.join(self.KITTI_velo, path)) and
               int(path) in scene_id
        ]

        print("list_of_scene: ", list_of_scene)

        # 遍历每一个序列列表中的序列
        for scene in list_of_scene:
            print("-" * 50)
            print("current scene id is: ", scene)
            # 标签路径
            label_file = os.path.join(self.KITTI_label, scene + ".txt")
            # 读取标签txt文件
            df = pd.read_csv(
                label_file,
                sep=' ',
                names=[
                    "frame", "track_id", "type", "truncated", "occluded",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "rotation_y"
                ])
            # df = df[df["type"] == self.category]  # 筛选出类别是car的标签

            # 筛选出类型为Car或者Van的标签行
            df = df[(df["type"] == "Car") | (df["type"] == "Van")]

            df.insert(loc=0, column="scene", value=scene)  # 在标签中插入一列表示这是哪个场景
            # 还原索引，将df中的数据的每一行的索引变成默认排序的形式
            df = df.reset_index(drop=True)
            length = df.shape[0]
            frame_id = -1
            last_frame_id = -1
            corners_ = []
            centers_ = []
            ids_ = []
            arr_bbox = BoundingBoxArray()
            
            try:
                for label_row in tqdm(range(length)):
                    # print(df.loc[0])
                    this_anno = df.loc[label_row]
                    last_frame_id = frame_id
                    frame_id = this_anno['frame']

                    bbox_msg = BoundingBox()
                    bbox_msg.header.stamp = rospy.Time(frame_id)
                    xmin, ymin, xmax, ymax = this_anno['bbox_left'], this_anno['bbox_top'], this_anno['bbox_right'], this_anno['bbox_bottom']

                    
                    bbox_msg.pose.position.x = xmin
                    bbox_msg.pose.position.y = ymin
                    bbox_msg.pose.position.z = 0  # 或者一个估计的深度值

                    # 设置盒子的尺寸
                    bbox_msg.dimensions.x = xmax - xmin
                    bbox_msg.dimensions.y = ymax - ymin
                    bbox_msg.dimensions.z = 0

                    bbox_msg.label = this_anno['track_id']
                    
                    arr_bbox.boxes.append(bbox_msg)
                    # print(frame_id, last_frame_id, frame_id != last_frame_id)
                    
                    if frame_id != last_frame_id :
                        arr_bbox.header.stamp = arr_bbox.boxes[0].header.stamp
                        last_box = arr_bbox.boxes[-1]
                        arr_bbox.boxes.pop()
                        self._bboxpub.publish(arr_bbox)

                        # 根据是否是第一次写入选择模式
                        mode = 'w' if self.is_first_write else 'a'
                        with rosbag.Bag(self.bag_filename, mode) as bag:
                            # 将点云消息写入bag文件
                            bag.write('/bbox_image_gt', arr_bbox, rospy.Duration(self.count * 0.1))
                        # print(this_anno["frame"]-1)
                        if (this_anno["frame"] - 1 >= 0) :
                            image_path = os.path.join(self.KITTI_img, this_anno["scene"],
                                            '%06d.png' % (this_anno["frame"] - 1))  # f'{box["frame"]:06}.bin')
                            scene = cv2.imread(image_path)
                            # label = f"ID: {this_anno['track_id']}"
                            for bbox in arr_bbox.boxes:
                                cv2.putText(scene, f"ID: {bbox.label}", (int(bbox.pose.position.x), int(bbox.pose.position.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 3)
                                cv2.rectangle(scene, (int(bbox.pose.position.x), int(bbox.pose.position.y)), (int(bbox.dimensions.x + bbox.pose.position.x), int(bbox.pose.position.y + bbox.dimensions.y)), (0, 255, 0), 2)
                            img_msg = self._bridge.cv2_to_imgmsg(scene, 'bgr8')
                            img_msg.header.stamp = rospy.Duration(self.count * 0.1)
                            self.imagepub.publish(img_msg)

                        arr_bbox.boxes = []
                        arr_bbox.boxes.append(last_box)
                        
                        

                        # if frame_id != last_frame_id or label_row == length - 1:
                        self.getImage(this_anno)
                        this_pc, this_box, state = self.getBBandPC(this_anno)  # this_pc's shape is (3, N)

                        if state is True:
                            points = this_pc.points.T
                            # --------------------- pub box to show label
                            corner_num = len(corners_)
                            if (corner_num) :
                                self.pub_box_markers(this_anno['frame'], np.array(corners_), np.array(ids_))
                                #     # Create a BoundingBox messasge
                                
                            corners_ = []
                            ids_ = []
                            corners_.append(np.concatenate(this_box.corners().transpose(), axis=0))
                            ids_.append(this_anno['track_id'])
                            # -------------------- pub box center
                            center_num = len(centers_)
                            if (center_num) :
                                self.pub_box_centers(this_anno['frame'], np.array(centers_))
                            centers_ = []
                            centers_.append(this_box.center)

                            # -------------------- save pcd
                            if self.save_pcd:
                                file_name = get_name_by_read_dir(pcd_path)
                                pc_save_pcd(points, pcd_path, file_name + '.pcd')
                            # -------------------- pub whole frame pc
                            self.publish_pointcloud(this_anno['frame'], points)
                            print("\n============================")
                            print(blue("scene: {} | frame: {}").format(this_anno['scene'], this_anno['frame']))
                            print(blue("pub pts with shape -> "), points.shape)
                            print(blue("pub markers with shape -> "), corner_num)
                            print(blue("pub centers with shape -> "), center_num)
                            # -------------------- sleeping
                            self.rate.sleep()
                        else:
                            print(red("Error! getBBandPC error"))
                    else:
                        _, this_box, state = self.getBBandPC(this_anno)  # this_pc's shape is (3, N)
                        # if state is True:
                            
                        centers_.append(this_box.center)
                        corners_.append(np.concatenate(this_box.corners().transpose(), axis=0).tolist())
                        ids_.append(this_anno['track_id'])
                        # else:
                        #     print(red("Error! getBBandPC error"))
                arr_bbox.header.stamp = arr_bbox.boxes[0].header.stamp
                # last_box = arr_bbox.boxes[-1]
                # arr_bbox.boxes.pop()
                self._bboxpub.publish(arr_bbox)
                # 根据是否是第一次写入选择模式
                mode = 'w' if self.is_first_write else 'a'
                with rosbag.Bag(self.bag_filename, mode) as bag:
                    # 将点云消息写入bag文件
                    bag.write('/bbox_image_gt', arr_bbox, rospy.Duration(self.count * 0.1))

                corner_num = len(corners_)
                if (corner_num) :
                    self.pub_box_markers(this_anno['frame'], np.array(corners_), np.array(ids_))

                label = f"ID: {this_anno['track_id']}"

                font_scale = 1.0  # 字体大小
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 2  # 文本线条厚度
            except KeyboardInterrupt:
                print("程序被用户中断")
                sys.exit(0)
                pass

    # 获取包含序列的列表
    def getSceneList(self, split):
        if "TRAIN" in split.upper():  # Training SET
            sceneID = list(range(0, 17))
        elif "VALID" in split.upper():  # Validation Set
            sceneID = list(range(17, 19))
        elif "TEST" in split.upper():  # Testing Set
            sceneID = list(range(19, 21))
        else:  # Full Dataset
            sceneID = list(range(21))
        # logging.info("sceneID_path:\n%s\n", sceneID)   
        return sceneID
    
    def getImage(self, anno):
        calib_path = os.path.join(self.KITTI_Folder, 'calib', anno['scene'] + ".txt")
        calib = self.read_calib_file(calib_path)
        # 在矩阵最下面叠加一行(0,0,0,1)
        transf_mat = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))

        try:
            # Image
            image_path = os.path.join(self.KITTI_img, anno["scene"],
                                         '%06d.png' % (anno["frame"]))  # f'{box["frame"]:06}.bin')
            # print(image_path)
            self.cv_image = cv2.imread(image_path)
            self.height, self.width = self.cv_image.shape[:2]
            
            img_msg = self._bridge.cv2_to_imgmsg(self.cv_image, 'bgr8')
            print(anno["frame"])
            img_msg.header.stamp = rospy.Time(anno["frame"])
            # self.imagepub.publish(img_msg)
            # 根据是否是第一次写入选择模式
            mode = 'w' if self.is_first_write else 'a'
            with rosbag.Bag(self.bag_filename, mode) as bag:
                # self.count += 1
                # 将点云消息写入bag文件
                bag.write('/kitti/camera_color_left/image_raw', img_msg, rospy.Duration(self.count * 0.1))
            
            # 第一次写入后更新标志
            if self.is_first_write:
                self.is_first_write = False

            # self.bag.write('/kitti/camera_color_left/image_raw', img_msg, rospy.Duration(self.count * 0.1))
                
        except FileNotFoundError:
            # logging.error("No such file found\n%s\n", velodyne_path)
            PC = PointCloud(np.array([[0, 0, 0]]).T)
            
    
    def getBBandPC(self, anno):
        calib_path = os.path.join(self.KITTI_Folder, 'calib', anno['scene'] + ".txt")
        calib = self.read_calib_file(calib_path)
        # 在矩阵最下面叠加一行(0,0,0,1)
        transf_mat = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))
        # print(transf_mat)
        PC, box, state = self.getPCandBBfromPandas(anno, transf_mat)
        return PC, box, state

    def getPCandBBfromPandas(self, box, calib):
        # 求出车辆的中心点 从此处的中心点是根据KITTI中相机坐标系下的中心点
        # 减去一半的高度移到地面上
        center = [box["x"], box["y"] - box["height"] / 2, box["z"]]
        size = [box["width"], box["length"], box["height"]]
        orientation = Quaternion(
            axis=[0, 1, 0], radians=box["rotation_y"])
        BB = Box(center, size, orientation)  # 用中心点坐标和w,h,l以及旋转角来初始化BOX这个类
        BB.inverse_transform(calib)
        
        State = True
        try:
            # VELODYNE PointCloud
            velodyne_path = os.path.join(self.KITTI_velo, box["scene"],
                                         '%06d.bin' % (box["frame"]))  # f'{box["frame"]:06}.bin')
            # print(velodyne_path)
            # 从点云的.bin文件中读取点云数据并且转换为4*x的矩阵，且去掉最后的一行的点云的密度表示数据
            PC = PointCloud(np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
            # 将点云转换到相机坐标系下　因为label中的坐标和h,w,l在相机坐标系下的
            # PC.transform(calib)
        except FileNotFoundError:
            # logging.error("No such file found\n%s\n", velodyne_path)
            PC = PointCloud(np.array([[0, 0, 0]]).T)
            State = False

        return PC, BB, State

    def publish_pointcloud(self, frame_num, pts):
        # pointcloud pub
        header = Header()
        header.stamp = rospy.Time(frame_num)
        header.frame_id = 'velodyne'
        cloud_msg = pc2.create_cloud_xyz32(header, pts)
        self.point_pub.publish(cloud_msg)

        # 根据是否是第一次写入选择模式
        mode = 'w' if self.is_first_write else 'a'
        with rosbag.Bag(self.bag_filename, mode) as bag:
            
            # 将点云消息写入bag文件
            bag.write('/kitti/velo/pointcloud', cloud_msg, rospy.Duration(self.count * 0.1))
            self.count += 1
        
        # 第一次写入后更新标志
        if self.is_first_write:
            self.is_first_write = False

    def pub_box_markers(self, frame_num, corners, ids):
        # all_bbox = MarkerArray()
        # for i, corner in enumerate(corners):
        #     point_0 = Point(corner[0], corner[1], corner[2])
        #     point_1 = Point(corner[3], corner[4], corner[5])
        #     point_2 = Point(corner[6], corner[7], corner[8])
        #     point_3 = Point(corner[9], corner[10], corner[11])
        #     point_4 = Point(corner[12], corner[13], corner[14])
        #     point_5 = Point(corner[15], corner[16], corner[17])
        #     point_6 = Point(corner[18], corner[19], corner[20])
        #     point_7 = Point(corner[21], corner[22], corner[23])

        #     marker = Marker(id=i)
        #     marker.type = Marker.LINE_LIST
        #     marker.ns = 'velodyne'
        #     marker.action = Marker.ADD
        #     marker.header.frame_id = "velodyne"
        #     marker.header.stamp = rospy.Time(frame_num)

        #     marker.points.append(point_1)
        #     marker.points.append(point_2)
        #     marker.points.append(point_1)
        #     marker.points.append(point_0)
        #     marker.points.append(point_1)
        #     marker.points.append(point_5)
        #     marker.points.append(point_7)
        #     marker.points.append(point_4)
        #     marker.points.append(point_7)
        #     marker.points.append(point_6)
        #     marker.points.append(point_7)
        #     marker.points.append(point_3)
        #     marker.points.append(point_2)
        #     marker.points.append(point_6)
        #     marker.points.append(point_2)
        #     marker.points.append(point_3)
        #     marker.points.append(point_0)
        #     marker.points.append(point_4)
        #     marker.points.append(point_0)
        #     marker.points.append(point_3)
        #     marker.points.append(point_5)
        #     marker.points.append(point_6)
        #     marker.points.append(point_5)
        #     marker.points.append(point_4)

        #     marker.lifetime = rospy.Duration.from_sec(0.2)
        #     marker.scale.x = 0.05
        #     marker.color.a = 1.0
        #     marker.color.r = 1.0
        #     marker.color.g = 0.0
        #     marker.color.b = 0.0
        #     marker.text = str(1)
        #     all_bbox.markers.append(marker)
        # self.box_marker_pub.publish(all_bbox)
        all_bbox = BoundingBoxArray()
        all_bbox_2d = BoundingBoxArray()
        all_bbox.header.frame_id = "velodyne"
        all_bbox.header.stamp = rospy.Time(frame_num)
        all_bbox_2d.header.frame_id = "velodyne"
        all_bbox_2d.header.stamp = rospy.Time(frame_num)

        for i, corner in enumerate(corners):
            bbox = BoundingBox()
            bbox.header.frame_id = "velodyne"
            bbox.header.stamp = rospy.Time(frame_num)

            # 计算边界框的中心点
            x_center = sum(corner[0::3]) / 8.0
            y_center = sum(corner[1::3]) / 8.0
            z_center = sum(corner[2::3]) / 8.0
            bbox.pose.position.x = x_center
            bbox.pose.position.y = y_center
            bbox.pose.position.z = z_center

            # 计算边界框的尺寸（这里简化处理，可能需要根据实际情况调整）
            x_size = max(corner[0::3]) - min(corner[0::3])
            y_size = max(corner[1::3]) - min(corner[1::3])
            z_size = max(corner[2::3]) - min(corner[2::3])
            bbox.dimensions.x = x_size
            bbox.dimensions.y = y_size
            bbox.dimensions.z = z_size

            # 设置边界框的颜色和其他属性（根据需要调整）
            bbox.value = ids[i]  # 可以用来标识边界框
            bbox.label = ids[i]  # 标签ID
            print("bbox.label: ", bbox.label)
            all_bbox.boxes.append(bbox)


            result = self.transform_and_project_bbox3d(bbox)
            if result is None:
                continue
            else:
                [xmin, ymin, xmax, ymax] = result
                # Create a BoundingBox message
                bbox = BoundingBox()

                bbox.header = bbox.header
                bbox.pose.position.x = xmin
                bbox.pose.position.y = ymin
                bbox.dimensions.x = xmax - xmin
                bbox.dimensions.y = ymax - ymin
                # bbox.value = scores[i]
                bbox.label = ids[i]
                # print(bbox.label)
                all_bbox_2d.boxes.append(bbox)
            
        # 根据是否是第一次写入选择模式
        mode = 'w' if self.is_first_write else 'a'
        with rosbag.Bag(self.bag_filename, mode) as bag:
            # 将点云消息写入bag文件
            bag.write('/bbox_lidar_3d', all_bbox, rospy.Duration(self.count * 0.1))
            bag.write('/bbox_lidar', all_bbox_2d, rospy.Duration(self.count * 0.1))
        self.lidar_pub.publish(all_bbox)
        self.lidar2d_pub.publish(all_bbox_2d)

        img = self.draw_bboxes_on_image(all_bbox_2d, self.cv_image)
        img_msg = self._bridge.cv2_to_imgmsg(img, 'bgr8')
        img_msg.header.stamp = rospy.Time(frame_num)
        self.projected_imagepub.publish(img_msg)

    def draw_bboxes_on_image(self, bbox_array, cv_image):
        """
        在OpenCV图像上绘制由BoundingBoxArray消息表示的2D边界框。

        参数:
        - bbox_array: BoundingBoxArray消息，包含要绘制的边界框。
        - cv_image: OpenCV格式的图像，边界框将被绘制在此图像上。
        """
        for bbox in bbox_array.boxes:
            # 边界框的2D坐标
            xmin = int(bbox.pose.position.x)
            ymin = int(bbox.pose.position.y)
            xmax = int(bbox.pose.position.x + bbox.dimensions.x)
            ymax = int(bbox.pose.position.y + bbox.dimensions.y)

            # 绘制边界框
            cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # 如果需要，可以在这里显示图像，或者返回修改后的图像
        # cv2.imshow('Image with BBoxes', cv_image)
        # cv2.waitKey(0)
        # 返回修改后的图像
        return cv_image

    def pub_box_centers(self, frame_num, centers):
        data = String()
        result = [frame_num] + np.concatenate(centers, axis=0).tolist()
        result = map(str, result)
        data = " ".join(result)
        self.box_center_pub.publish(data)

    @staticmethod
    def read_calib_file(filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    pass
        # 返回一个字典　字典中有6个键对　每个键对应的是calib文件中的一行，
        # key是'P0'，value是后面的对应的表示数值转换的一个3*4的numpy矩阵
        return data

    @staticmethod
    def make_sure_path_valid(dirs):
        if not os.path.exists(dirs):
            os.makedirs(dirs)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Generate the pcd and label.txt file of fixed sequence in KITTI',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--category', required=False, type=str,
                        default='Car', help='category_name Car/Pedestrian/Van/Cyclist')
    parser.add_argument('--dataset_path', required=False, type=str,
                        default='/media/echo/仓库卷/DataSet/Autonomous_Driving/KITTI/tracking/origin_dataset/training',
                        help='dataset Path')
    parser.add_argument('--save_path', required=False, type=str,
                        default='saved',
                        help='save Path')
    parser.add_argument('--replace', required=False, type=bool,
                        default=True, help='whether delete the all files and generate again or not')
    parser.add_argument('--save_pcd', required=False, type=bool,
                        default=False, help='whether save whole frame pointcloud data as .pcd or not')
    parser.add_argument('--pubRate', required=False, type=int,
                        default=10, help='The rate of topic publish in ros. /Hz')

    args = parser.parse_args()
    kitti = RosVisualizer(args)
    scene_list = kitti.get_sceneID()
    kitti.pub_pc_and_box(scene_list)
