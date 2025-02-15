import numpy as np
import cv2 as cv
from glob import glob
from feature_extraction import FeatureExtraction
from radar_data import *


def show_bscope_features(transformation: FeatureExtraction):
    color = np.random.randint(0, 255, (100, 3))
    for i, (point_old, point_new) in enumerate(
        zip(transformation.radar_point_old, transformation.radar_point_new)
    ):
        residual = np.sum(
            np.abs(
                np.subtract(
                    transform_point(
                        point_old.x, point_old.y, transformation.affine_matrix
                    ),
                    [point_new.x, point_new.y],
                )
            )
        )
        if residual < 10.0:
            new_frame = cv.circle(
                transformation.new_frame,
                (int(point_new.pixel_x), int(point_new.pixel_y)),
                5,
                color[i].tolist(),
                -1,
            )
            transformed_point = transform_point(
                point_old.x, point_old.y, transformation.affine_matrix
            )
            pixel_coor = get_bscope_pixel_coordinate(
                transformed_point[0], transformed_point[1]
            )
            old_frame = cv.circle(
                transformation.old_frame,
                (point_old.pixel_x, point_old.pixel_y),
                5,
                color[i].tolist(),
                -1,
            )
            test_frame = cv.circle(
                transformation.test_frame,
                (pixel_coor[0], pixel_coor[1]),
                5,
                color[i].tolist(),
                -1,
            )
    return old_frame, new_frame, test_frame


def show_radar_image_features(transformation: FeatureExtraction):
    color = np.random.randint(0, 255, (100, 3))
    test_frame = copy.deepcopy(transformation.radar_data_new.radar_image)
    old_frame = copy.deepcopy(transformation.radar_data_old.radar_image)
    new_frame = copy.deepcopy(transformation.radar_data_new.radar_image)
    for i, (point_old, point_new) in enumerate(
        zip(transformation.radar_point_old, transformation.radar_point_new)
    ):
        radar_image_coordinate_new = get_radar_image_cartesian_coordinate(
            point_new.x, point_new.y
        )
        new_frame = cv.circle(
            new_frame,
            (radar_image_coordinate_new[0], radar_image_coordinate_new[1]),
            5,
            color[i].tolist(),
            -1,
        )
        transformed_point = transform_point(
            point_old.x, point_old.y, transformation.affine_matrix
        )
        transformed_image_coor = get_radar_image_cartesian_coordinate(
            transformed_point[0], transformed_point[1]
        )
        radar_image_coordinate_old = get_radar_image_cartesian_coordinate(
            point_old.x, point_old.y
        )
        old_frame = cv.circle(
            old_frame,
            (radar_image_coordinate_old[0], radar_image_coordinate_old[1]),
            5,
            color[i].tolist(),
            -1,
        )
        test_frame = cv.circle(
            test_frame,
            (transformed_image_coor[0], transformed_image_coor[1]),
            5,
            color[i].tolist(),
            -1,
        )
    return old_frame, new_frame, test_frame
