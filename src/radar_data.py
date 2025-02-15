import scipy.io
import pandas
import cv2 as cv
import math
import numpy as np
import matplotlib.pyplot as plt
import copy

"""
Parameters for the transformation between bscope image pixel coordinates and sensor coordinate.
"""
# The maximum detection range of radar data
MAX_RANGE = 25.0
# The maximum detection range of radar data
MIN_RANGE = 5.0
# The minimum azimuth angle of radar data
MIN_AZIMUTH_ANGLE = -45.0
# The maximum azimuth angle of radar data
MAX_AZIMUTH_ANGLE = 45.0
# The channel number of azimuth angles
AZIMUTH_CHANNEL_NUM = 199
# The range profile sample number over detection range.
RANGE_PROFILE_SAMPLE = 652

"""
Parameters for radar image in cartesian coordinate.
"""
# The maximum pixel width of radar image
MAX_X = 545
# The maximum pixel height of radar image
MAX_Y = 328
# The origin pixel x point correspond to 0 azimuth angle.
ORIGINAL_Y = 386
# The origin pixel y point correspond to minimum detection range.
ORIGINAL_X = 273
# The scale resolution between radar pixel to real work distance
SCALE = 15.44

"""
The radar point data type.


Returns:
    _type_: _description_
"""
class RadarPoint:
    def __init__(
        self, intensity: float, pixel: int, pixel_x: int, pixel_y: int
    ) -> None:
        self.intensity = intensity
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.azimuth = (
            (pixel_x / AZIMUTH_CHANNEL_NUM) * (MAX_AZIMUTH_ANGLE - MIN_AZIMUTH_ANGLE)
        ) - MAX_AZIMUTH_ANGLE
        self.range = ((MAX_RANGE - MIN_RANGE) / RANGE_PROFILE_SAMPLE) * (
            RANGE_PROFILE_SAMPLE - pixel_y
        ) + MIN_RANGE

        # sensor coordinate with sensor position as the original point.
        self.x = self.range * math.sin(math.radians(self.azimuth))
        self.y = self.range * math.cos(math.radians(self.azimuth))
        self.pixel = pixel

        # Define the class label number of each radar cell. 0: unclassified, 1: shadow, 2: asphalt, 3: grass, 4: objects
        self.class_label = 0
        self.coordinate = np.array([self.x, self.y, 0])
        self.image_pixels = []


class RadarData:
    def __init__(
        self,
        bscope_folder: str,
        image_folder: str,
        label_folder: str,
        data_folder: str,
        frame_num: int,
    ) -> None:
        mat = scipy.io.loadmat(data_folder + "rawImg_scan_" + str(frame_num) + ".mat")
        self.data = mat.get("currentScan")
        self.bscope = cv.imread(
            bscope_folder + "img_scan_" + str(f"{frame_num:03}") + ".png",
            cv.IMREAD_COLOR,
        )
        self.radar_image = cv.imread(
            image_folder + "img_scan_" + str(f"{frame_num:03}") + ".png",
            cv.IMREAD_COLOR,
        )
        self.radar_points = []
        for c_x in range(0, AZIMUTH_CHANNEL_NUM):
            column = []
            for c_y in range(0, RANGE_PROFILE_SAMPLE):
                point = RadarPoint(
                    self.data[RANGE_PROFILE_SAMPLE - c_y][c_x],
                    self.bscope[c_y][c_x],
                    c_x,
                    c_y,
                )
                column.append(point)
            self.radar_points.append(column)
        for c_x in range(0, MAX_X):
            for c_y in range(0, MAX_Y):
                coordinate = get_radar_cell_from_radar_image_pixel(c_x, c_y)
                self.radar_points[coordinate[0]][coordinate[1]].image_pixels.append(
                    [c_x, c_y]
                )
        self.label_image = cv.imread(
            label_folder + "img_scan_" + str(f"{frame_num:03}") + ".png",
            cv.IMREAD_COLOR,
        )


###### The related coordinate transformation methods between radar image, radar cells and sensor coordinate. #####


# Compute the radar cell coordinate according to radar image pixel coordinate
def get_radar_cell_from_radar_image_pixel(pixel_image_x, pixel_image_y):
    x_coordinate = (pixel_image_x - ORIGINAL_X) / SCALE
    y_coordinate = (ORIGINAL_Y - pixel_image_y) / SCALE
    pixel_x = int(
        (math.degrees(math.atan(x_coordinate / y_coordinate)) + MAX_AZIMUTH_ANGLE)
        / (MAX_AZIMUTH_ANGLE - MIN_AZIMUTH_ANGLE)
        * AZIMUTH_CHANNEL_NUM
    )
    pixel_y = int(
        RANGE_PROFILE_SAMPLE
        - (
            (math.sqrt(x_coordinate**2 + y_coordinate**2) - MIN_RANGE)
            * RANGE_PROFILE_SAMPLE
        )
        / (MAX_RANGE - MIN_RANGE)
    )
    pixel_x = max(0, min(pixel_x, AZIMUTH_CHANNEL_NUM - 1))
    pixel_y = max(0, min(pixel_y, RANGE_PROFILE_SAMPLE - 1))
    return [pixel_x, pixel_y]


# Transform a 2d point according to input x, y and affine matrix
def transform_point(x, y, affine_matrix):
    R = affine_matrix[:, :2]
    t = affine_matrix[:, 2]
    transformed_point = np.dot(R, np.array([x, y]).T) + t.T
    return transformed_point[0], transformed_point[1]


# Compute the bscope coordinate according to the input sensor coordinate.
def get_bscope_pixel_coordinate(x_coordinate=None, y_coordinate=None):
    pixel_x = int(
        (math.degrees(math.atan(x_coordinate / y_coordinate)) + MAX_AZIMUTH_ANGLE)
        / (MAX_AZIMUTH_ANGLE - MIN_AZIMUTH_ANGLE)
        * AZIMUTH_CHANNEL_NUM
    )
    pixel_y = int(
        RANGE_PROFILE_SAMPLE
        - (
            (math.sqrt(x_coordinate**2 + y_coordinate**2) - MIN_RANGE)
            * RANGE_PROFILE_SAMPLE
        )
        / (MAX_RANGE - MIN_RANGE)
    )
    pixel_x = max(0, min(pixel_x, AZIMUTH_CHANNEL_NUM - 1))
    pixel_y = max(0, min(pixel_y, RANGE_PROFILE_SAMPLE - 1))
    return [pixel_x, pixel_y]


# Compute the pixel coordinate of radar image according to the input coordinate in sensor coordinate.
def get_radar_image_cartesian_coordinate(x_coordinate, y_coordinate):
    x_pixel_dis = x_coordinate * SCALE
    y_pixel_dis = y_coordinate * SCALE
    pixel_x = max(0, min(MAX_X - 1, int(x_pixel_dis + ORIGINAL_X)))
    pixel_y = max(0, min(MAX_Y - 1, int(ORIGINAL_Y - y_pixel_dis)))
    return [pixel_x, pixel_y]


# Compute the sensor coordinate according to radar image pixel coordinate
def get_sensor_coordinate_from_radar_image_pixel(pixel_image_x, pixel_image_y):
    x_coordinate = (pixel_image_x - ORIGINAL_X) / SCALE
    y_coordinate = (ORIGINAL_Y - pixel_image_y) / SCALE
    return x_coordinate, y_coordinate
