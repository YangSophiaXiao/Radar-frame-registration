import numpy as np
import cv2 as cv
from glob import glob
import math
from radar_data import *
import copy
from scipy.optimize import minimize

"""
The class of radar frame registration with methods of SIFT feature detection, 
feature matching, and non-linear estimation on RANSAC.

Args:
    radar_data_old: The previous radar frame
    radar_data_new: The current radar frame
    use_radar_image: The flag to choose radar cell data or radar image data
Returns:
    _type_: _description_
"""
class FeatureExtraction:
    def __init__(
        self,
        radar_data_old: RadarData,
        radar_data_new: RadarData,
        use_radar_image=False,
    ) -> None:
        self.radar_data_old = radar_data_old
        self.radar_data_new = radar_data_new
        self.use_radar_image = use_radar_image
        if self.use_radar_image:
            self.old_frame = copy.deepcopy(self.radar_data_old.radar_image)
            self.new_frame = copy.deepcopy(self.radar_data_new.radar_image)
            self.test_frame = copy.deepcopy(self.radar_data_new.radar_image)
        else:
            self.old_frame = copy.deepcopy(self.radar_data_old.bscope)
            self.new_frame = copy.deepcopy(self.radar_data_new.bscope)
            self.test_frame = copy.deepcopy(self.radar_data_new.bscope)

        self.good_new = 0
        self.good_old = 0
        self.radar_point_old = []
        self.radar_point_new = []
        self.r = np.identity(2)
        self.t = np.zeros([2, 1])
        self.affine_matrix = np.zeros([2, 3])
        self.affine_matrix_radar_image = np.zeros([2, 3])
        self.matches_img = None

    """
    Extract the good feature corresponding points based on SIFT and match them using FLANN-based matcher.
    """
    def sift_feature_extraction(self):
        sift = cv.SIFT_create()
        # Detect keypoints and compute descriptors
        frame1 = cv.cvtColor(self.old_frame, cv.COLOR_BGR2GRAY)
        frame2 = cv.cvtColor(self.new_frame, cv.COLOR_BGR2GRAY)

        keypoints_old, descriptors_old = sift.detectAndCompute(frame1, None)
        keypoints_new, descriptors_new = sift.detectAndCompute(frame2, None)

        # Match descriptors using FLANN-based matcher
        index_params = dict(algorithm=1, trees=10)  # FLANN with KDTree
        search_params = dict(checks=80)  # Number of checks
        matcher = cv.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(descriptors_old, descriptors_new, k=2)

        # Apply Lowe's ratio test to filter matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Draw matches
        self.matches_img = self.draw_matches_custom(
            self.old_frame,
            keypoints_old,
            self.new_frame,
            keypoints_new,
            good_matches[0:20],
        )

        # Extract matched keypoints
        if len(good_matches) > 4:  # Minimum points needed for homography
            self.good_old = np.float32(
                [keypoints_old[m.queryIdx].pt for m in good_matches]
            )
            self.good_new = np.float32(
                [keypoints_new[m.trainIdx].pt for m in good_matches]
            )
            
        # Estimate the affine transformation matrix estimated by radar image pixels
        if self.use_radar_image:
            self.affine_matrix_radar_image, inliers = cv.estimateAffine2D(
                self.good_old,
                self.good_new,
                method=cv.RANSAC,
                ransacReprojThreshold=1.0,
            )
        else:
            radar_image_old = []
            radar_image_new = []
            for i, (bscope_old, bscope_new) in enumerate(
                zip(self.good_old, self.good_new)
            ):
                point_old = self.radar_data_old.radar_points[math.floor(bscope_old[0])][
                    math.floor(bscope_old[1])
                ]
                radar_image_old.append(
                    get_radar_image_cartesian_coordinate(point_old.x, point_old.y)
                )
                point_new = self.radar_data_new.radar_points[math.floor(bscope_new[0])][
                    math.floor(bscope_new[1])
                ]
                radar_image_new.append(
                    get_radar_image_cartesian_coordinate(point_new.x, point_new.y)
                )
            radar_image_old = np.float32(radar_image_old)
            radar_image_new = np.float32(radar_image_new)
            self.affine_matrix_radar_image, inliers = cv.estimateAffine2D(
                radar_image_old, radar_image_new, method=cv.RANSAC
            )

        # Estimate the affine matrix in sensor coordinate space
        self.solve_affine_2d()
        

    """
    The method to compute affine matrix based on the extracted features and RANSAC.
    """
    def solve_affine_2d(self):
        # Extract the key points in cartesian sensor cooßrdinate.
        x_list = []
        y_list = []
        if self.use_radar_image:
            for i, (x, y) in enumerate(zip(self.good_old, self.good_new)):
                if (x[0] < MAX_X and x[1] < MAX_Y) and (y[0] < MAX_X and y[1] < MAX_Y):
                    ## Get the feature radar points in old frame
                    coordinate_old = get_sensor_coordinate_from_radar_image_pixel(
                        x[0], x[1]
                    )
                    bscope_pixle_old = get_bscope_pixel_coordinate(
                        coordinate_old[0], coordinate_old[1]
                    )
                    point_old = self.radar_data_old.radar_points[
                        math.floor(bscope_pixle_old[0])
                    ][math.floor(bscope_pixle_old[1])]
                    self.radar_point_old.append(point_old)
                    ## Get the feature radar points in new frame
                    coordinate_new = get_sensor_coordinate_from_radar_image_pixel(
                        y[0], y[1]
                    )
                    bscope_pixle_new = get_bscope_pixel_coordinate(
                        coordinate_new[0], coordinate_new[1]
                    )
                    point_new = self.radar_data_new.radar_points[
                        math.floor(bscope_pixle_new[0])
                    ][math.floor(bscope_pixle_new[1])]
                    self.radar_point_new.append(point_new)
        else:
            for i, (x, y) in enumerate(zip(self.good_old, self.good_new)):
                if (x[0] < AZIMUTH_CHANNEL_NUM and x[1] < RANGE_PROFILE_SAMPLE) and (
                    y[0] < AZIMUTH_CHANNEL_NUM and y[1] < RANGE_PROFILE_SAMPLE
                ):
                    point_old = self.radar_data_old.radar_points[math.floor(x[0])][
                        math.floor(x[1])
                    ]
                    self.radar_point_old.append(point_old)
                    point_new = self.radar_data_new.radar_points[math.floor(y[0])][
                        math.floor(y[1])
                    ]
                    self.radar_point_new.append(point_new)
        ## Extract the x and y coordinates for affine transformation optimization.
        x_list = np.array(
            [[point_x.x, point_x.y] for point_x in self.radar_point_old]
        )  
        y_list = np.array(
            [[point_y.x, point_y.y] for point_y in self.radar_point_new]
        )  
        
        # Estimate the affine transformation in 2d based on RANSAC.
        self.affine_matrix, inliers = cv.estimateAffine2D(
            np.float32(x_list), np.float32(y_list), method=cv.RANSAC
        )
        self.r = self.affine_matrix[:, :2]
        self.t = self.affine_matrix[:, 2]
        
    """
    The method to draw the matched feature points.
    """        
    def draw_matches_custom(
        self,
        img1,
        keypoints1,
        img2,
        keypoints2,
        matches,
        keypoint_color=(0, 0, 255),  # Green
        line_color=(255, 255, 255),  # Blue
        keypoint_size=7,
        line_thickness=2,
    ):
        # Create a combined image to draw matches
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]
        combined_img = np.zeros(
            (max(height1, height2), width1 + width2, 3), dtype=np.uint8
        )
        combined_img[:height1, :width1] = (
            img1 if len(img1.shape) == 3 else cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
        )
        combined_img[:height2, width1:] = (
            img2 if len(img2.shape) == 3 else cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
        )

        # Draw each match
        for match in matches:
            # Get the matching keypoints
            kp1 = keypoints1[match.queryIdx].pt  # (x, y) in img1
            kp2 = keypoints2[match.trainIdx].pt  # (x, y) in img2

            # Adjust kp2's x-coordinate to align with img2 in the combined image
            kp2 = (kp2[0] + width1, kp2[1])

            # Draw keypoints as circles
            cv.circle(
                combined_img,
                (int(kp1[0]), int(kp1[1])),
                keypoint_size,
                keypoint_color,
                -1,
            )
            cv.circle(
                combined_img,
                (int(kp2[0]), int(kp2[1])),
                keypoint_size,
                keypoint_color,
                -1,
            )

            # Draw a line connecting the keypoints
            cv.line(
                combined_img,
                (int(kp1[0]), int(kp1[1])),
                (int(kp2[0]), int(kp2[1])),
                line_color,
                line_thickness,
            )

        return combined_img

    """
    The helper method to transform points using affine matrix.
    """
    def transform_points(self, point):
        transformed_point = np.dot(self.r, np.array(point).T) + self.t.T
        return transformed_point
