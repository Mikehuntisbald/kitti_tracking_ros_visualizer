# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2018.
# Licensed under the Creative Commons [see licence.txt]

import numpy as np
from pyquaternion import Quaternion


class PointCloud:
    def __init__(self, points):
        """
        Class for manipulating and viewing point clouds.
        :param points: <np.float: 4, n>. Input point cloud matrix.
        """
        self.points = points
        # print("PC shape: ", self.points.shape)

        if self.points.shape[0] > 3:
            self.points = self.points[0:3, :]

    @staticmethod
    def load_pcd_bin(file_name):
        """
        Loads from binary format. Data is stored as (x, y, z, intensity, ring index).
        :param file_name: <str>.
        :return: <np.float: 4, n>. Point cloud matrix (x, y, z, intensity).
        """
        scan = np.fromfile(file_name, dtype=np.float32)
        points = scan.reshape((-1, 5))[:, :4]
        return points.T

    @classmethod
    def from_file(cls, file_name):
        """
        Instantiate from a .pcl, .pdc, .npy, or .bin file.
        :param file_name: <str>. Path of the pointcloud file on disk.
        :return: <PointCloud>.
        """

        if file_name.endswith('.bin'):
            points = cls.load_pcd_bin(file_name)
        elif file_name.endswith('.npy'):
            points = np.load(file_name)
        else:
            raise ValueError('Unsupported filetype {}'.format(file_name))

        return cls(points)

    def nbr_points(self):
        """
        Returns the number of points.
        :return: <int>. Number of points.
        """
        return self.points.shape[1]

    def subsample(self, ratio):
        """
        Sub-samples the pointcloud.
        :param ratio: <float>. Fraction to keep.
        :return: <None>.
        """
        selected_ind = np.random.choice(np.arange(0, self.nbr_points()),
                                        size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def remove_close(self, radius):
        """
        Removes point too close within a certain radius from origin.
        :param radius: <float>.
        :return: <None>.
        """

        x_filt = np.abs(self.points[0, :]) < radius
        y_filt = np.abs(self.points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        self.points = self.points[:, not_close]

    def translate(self, x):
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        :return: <None>.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_matrix):
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        :return: <None>.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def transform(self, transf_matrix):
        """
        Applies a homogeneous transform.
        :param transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
        :return: <None>.
        """
        self.points[:3, :] = transf_matrix.dot(
            np.vstack((self.points[:3, :], np.ones(self.nbr_points()))))[:3, :]

    def inverse_transform(self, transf_matrix):
        """
        Applies the inverse of a homogeneous transform.
        :param transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
        :return: <None>.
        """
        # 计算逆变换矩阵
        inv_transf_matrix = np.linalg.inv(transf_matrix)
        
        # 应用逆变换矩阵
        # 首先将点的坐标扩展到四维齐次坐标
        homogeneous_points = np.vstack((self.points[:3, :], np.ones(self.nbr_points())))
        # 然后通过逆变换矩阵应用逆变换
        transformed_points = inv_transf_matrix.dot(homogeneous_points)
        # 只取变换后的前三行，更新点的坐标
        self.points[:3, :] = transformed_points[:3, :]

    @staticmethod
    def fromPytorch(cls, pytorchTensor):
        """
        Loads from binary format. Data is stored as (x, y, z, intensity, ring index).
        :param pyttorchTensor: <Tensor>.
        :return: <np.float: 4, n>. Point cloud matrix (x, y, z, intensity).
        """
        points = pytorchTensor.numpy()
        # points = points.reshape((-1, 5))[:, :4]
        return cls(points)

    def normalize(self, wlh):
        normalizer = [wlh[1], wlh[0], wlh[2]]
        self.points = self.points / np.atleast_2d(normalizer).T


class Box:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self, center, size, orientation, label=np.nan, score=np.nan, velocity=(np.nan, np.nan, np.nan),
                 name=None):
        """
        :param center: [<float>: 3]. Center of box given as x, y, z.
        :param size: [<float>: 3]. Size of box in width, length, height.
        :param orientation: <Quaternion>. Box orientation.
        :param label: <int>. Integer label, optional.
        :param score: <float>. Classification score, optional.
        :param velocity: [<float>: 3]. Box velocity in x, y, z direction.
        :param name: <str>. Box name, optional. Can be used e.g. for denote category name.
        """
        assert not np.any(np.isnan(center))  # np.any()函数用于判断center里面只要不是全为空值　即返回True 是为了保证center里面没有空值
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label  # label里面有数的话转成int类型
        # 没有数值的话　　也就是说是nan的话　里面就是nan
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or
               (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'vel: {:.2f}, {:.2f}, {:.2f}, name: {}'

        return repr_str.format(self.label, self.score, self.center[0], self.center[1], self.center[2], self.wlh[0],
                               self.wlh[1], self.wlh[2], self.orientation.axis[0], self.orientation.axis[1],
                               self.orientation.axis[2], self.orientation.degrees, self.orientation.radians,
                               self.velocity[0], self.velocity[1], self.velocity[2], self.name)

    def encode(self):
        """
        Encodes the box instance to a JSON-friendly vector representation.
        :return: [<float>: 16]. List of floats encoding the box.
        """
        return self.center.tolist() + self.wlh.tolist() + self.orientation.elements.tolist() + [self.label] + [
            self.score] + self.velocity.tolist() + [self.name]

    @classmethod
    def decode(cls, data):
        """
        Instantiates a Box instance from encoded vector representation.
        :param data: [<float>: 16]. Output from encode.
        :return: <Box>.
        """
        return Box(data[0:3], data[3:6], Quaternion(data[6:10]), label=data[10], score=data[11], velocity=data[12:15],
                   name=data[15])

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: (3, 3)>.
        """
        return self.orientation.rotation_matrix

    def translate(self, x):
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        :return: <None>.
        """
        self.center += x

    def rotate(self, quaternion: Quaternion):
        """
        Rotates box.
        :param quaternion: <Quaternion>. Rotation to apply.
        :return: <None>.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def transform(self, transf_matrix):
        transformed = np.dot(transf_matrix[0:3, 0:4].T, self.center)
        self.center = transformed[0:3] / transformed[3]
        self.orientation = self.orientation * Quaternion(matrix=transf_matrix[0:3, 0:3])
        self.velocity = np.dot(transf_matrix[0:3, 0:3], self.velocity)

    def inverse_transform(self, transf_matrix):
        # 逆旋转矩阵
        inv_rot_matrix = np.linalg.inv(transf_matrix[0:3, 0:3])
        
        # 1. 逆变换速度
        self.velocity = np.dot(inv_rot_matrix, self.velocity)
        
        # 2. 逆变换方向
        inv_orientation = Quaternion(matrix=inv_rot_matrix)
        self.orientation = inv_orientation * self.orientation
        
        # 3. 构建完整的逆变换矩阵
        # Perform the inverse transformation
        inv_transf_matrix = np.linalg.inv(transf_matrix)
        transformed_center = np.dot(inv_transf_matrix, np.append(self.center, 1))
        
        # Now, transformed_center should have 4 elements, including the homogeneous coordinate
        # Update self.center with the transformed position, normalized by the homogeneous coordinate
        self.center = transformed_center[0:3] / transformed_center[3]
        
    def corners(self, wlh_factor: float = 1.0):
        """
        Returns the bounding box corners.
        :param wlh_factor: <float>. Multiply w, l, h by a factor to inflate or deflate the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self):
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]
