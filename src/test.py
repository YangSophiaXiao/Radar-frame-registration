from radar_data import *
from feature_extraction import *
import matplotlib.pyplot as plt
import pandas as pd
import os
from free_methods import *

# Define the input radar image type, false with radar cell input, true with radar image input
USE_RADAR_IMAGE_FLAG = False
# Flag for getting affine matrix in radar image space.
USE_RADAR_IMAGE_AFFINE_MATRIX = 1
# Flag for getting affine matrix in sensor coordinate.
USE_SENSOR_COORDINATE_AFFINE_MATRIX = 2
AFFINE_MATRIX_TYPE = USE_RADAR_IMAGE_AFFINE_MATRIX

## flags for debugging data output
SAVE_FEATURE_POINTS_RADAR_IMAGE = False
SAVE_FEATURE_POINTS_BSCOPE = False
SAVE_LABEL_OVERLAP = True
SAVE_RESULTS = True

# The color coding of label images
classification_color = {
    "asphalt": [255, 100, 0],
    "grass": [100, 200, 0],
    "shadows": [150, 150, 150],
    "targets": [0, 200, 200],
    "non_classify": [255, 255, 255],
}

# Visualize the overlap on lable images according to the sensor coordinate affine matrices.
def label_overlap(
    class_type: str,
    radar_data_old,
    radar_data_new,
    transformation,
    affine_matrix_flag: int,
):
    label_image_old = radar_data_old.label_image
    label_image_new = radar_data_new.label_image
    color = classification_color[class_type]
    pixels = np.where(
        (label_image_old[:, :, 0] == color[0])
        & (label_image_old[:, :, 1] == color[1])
        & (label_image_old[:, :, 2] == color[2])
    )
    image_pixels = []
    for c_y, c_x in zip(pixels[0], pixels[1]):
        if affine_matrix_flag == USE_SENSOR_COORDINATE_AFFINE_MATRIX:
            x_coordinate, y_coordinate = get_sensor_coordinate_from_radar_image_pixel(
                c_x, c_y
            )
            x_transformed, y_transformed = transform_point(
                x_coordinate, y_coordinate, transformation.affine_matrix
            )
            image_pixel = get_radar_image_cartesian_coordinate(
                x_transformed, y_transformed
            )
        elif affine_matrix_flag == USE_RADAR_IMAGE_AFFINE_MATRIX:
            x_coordinate, y_coordinate = c_x, c_y
            x_transformed, y_transformed = transform_point(
                x_coordinate, y_coordinate, transformation.affine_matrix_radar_image
            )
            x_transformed = max(0, min(MAX_X - 1, int(x_transformed)))
            y_transformed = max(0, min(MAX_Y - 1, int(y_transformed)))
            image_pixel = [x_transformed, y_transformed]
        image_pixels.append(image_pixel)

    mask = np.zeros_like(label_image_old)
    for pixel in image_pixels:
        mask[pixel[1], pixel[0], :] = [0, 0, 255]
    canny = cv.Canny(mask, 110, 150)
    # dilate the detected canny edge
    canny_dilate = cv.dilate(canny, None, iterations=2)
    ###processing the contours###
    contours, hierarchy = cv.findContours(
        canny_dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )
    overlap_image = cv.drawContours(label_image_new, contours, -1, (0, 255, 0), 3)
    return label_image_old, label_image_new, overlap_image

def overlap_radar_image_pixels(radar_image_old, radar_image_new, affine_matrix):
    threshold = 20
    radar_image_old_gray = cv.cvtColor(radar_image_old, cv.COLOR_BGR2GRAY)
    radar_image_new_gray = cv.cvtColor(radar_image_new, cv.COLOR_BGR2GRAY)
    # Warp the second image using the affine transformation matrix
    height, width = radar_image_new_gray.shape
    aligned_img = cv.warpAffine(radar_image_old, affine_matrix, (width, height))
    black_mask = np.all(aligned_img == [0, 0, 0], axis=-1)
    # Find coordinates of white pixels
    black_pixel_coords = np.argwhere(black_mask)
    for pixel in black_pixel_coords:
        aligned_img[pixel[0], pixel[1]] = [255, 255, 255]
    aligned_img_gray = cv.cvtColor(aligned_img, cv.COLOR_BGR2GRAY)
    # Subtract the images
    difference = cv.absdiff(aligned_img_gray, radar_image_new_gray)
    original_diff = cv.absdiff(radar_image_old_gray, radar_image_new_gray)
    # Remove the pixels which are in the white margin of both old and new images
    white_pixels = np.argwhere(
        (aligned_img_gray == 255) | (radar_image_new_gray == 255)
    )
    for pixel in white_pixels:
        difference[pixel[0], pixel[1]] = 0
        original_diff[pixel[0], pixel[1]] = 0
    # Threshold the difference image
    _, thresh = cv.threshold(difference, threshold, 255, cv.THRESH_BINARY)
    _, thresh_original = cv.threshold(original_diff, threshold, 255, cv.THRESH_BINARY)
    # Calculate overlap percentage
    overlap = (
        (np.sum(thresh == 0) - white_pixels.shape[0])
        / (thresh.size - white_pixels.shape[0])
        * 100
    )
    overlap_original = (
        (np.sum(thresh_original == 0) - white_pixels.shape[0])
        / (thresh_original.size - white_pixels.shape[0])
        * 100
    )
    residuals_pixel_intensity = np.sum(difference) / (
        thresh.size - white_pixels.shape[0]
    )
    residuals_pixel_intensity_original = np.sum(original_diff) / (
        thresh_original.size - white_pixels.shape[0]
    )
    return (
        overlap,
        residuals_pixel_intensity,
        overlap_original,
        residuals_pixel_intensity_original,
    )

def sift_evaluation(
    debug_path: str,
    frame_new: int,
    frame_old=None,
    save_feature_points_radar_image=False,
    save_feature_points_bscope=False,
    save_label_overlap=False,
    save_results=True,
):
    if frame_old == None:
        frame_old = frame_new + 1
    radar_data_old = RadarData(
        bscope_folder, image_folder, label_folder, data_folder, frame_old
    )
    radar_data_new = RadarData(
        bscope_folder, image_folder, label_folder, data_folder, frame_new
    )
    transformation = FeatureExtraction(
        radar_data_old, radar_data_new, USE_RADAR_IMAGE_FLAG
    )
    transformation.sift_feature_extraction()
    if save_feature_points_radar_image:
        old, new, transformed = show_radar_image_features(transformation)
        cv.imwrite(
            debug_path
            + "radar_image_sift_"
            + str(f"{frame_new:03}")
            + "_frame_old"
            + ".png",
            old,
        )
        cv.imwrite(
            debug_path
            + "radar_image_sift_"
            + str(f"{frame_new:03}")
            + "_frame_new"
            + ".png",
            new,
        )
        cv.imwrite(
            debug_path
            + "radar_image_sift_"
            + str(f"{frame_new:03}")
            + "_"
            + str(f"{frame_old:03}")
            + "_frame_transformed"
            + ".png",
            transformed,
        )
    if save_feature_points_bscope:
        old, new, transformed = show_bscope_features(transformation)
        cv.imwrite(
            debug_path
            + "bscope_sift_"
            + str(f"{frame_new:03}")
            + "_frame_old"
            + ".png",
            old,
        )
        cv.imwrite(
            debug_path
            + "bscope_sift_"
            + str(f"{frame_new:03}")
            + "_frame_new"
            + ".png",
            new,
        )
        cv.imwrite(
            debug_path
            + "bscope_sift_"
            + str(f"{frame_new:03}")
            + "_"
            + str(f"{frame_old:03}")
            + "_frame_transformed"
            + ".png",
            transformed,
        )
    if save_label_overlap:
        old, new, transformed = label_overlap(
            "targets",
            radar_data_old,
            radar_data_new,
            transformation,
            AFFINE_MATRIX_TYPE,
        )
        cv.imwrite(
            debug_path
            + "label_sift_"
            + str(f"{frame_new:03}")
            + "_"
            + str(f"{frame_old:03}")
            + "_frame_transformed"
            + ".png",
            transformed,
        )
    if save_results:
        (
            overlap,
            pixel_diff_intensity,
            overlap_original,
            pixel_diff_intensity_original,
        ) = overlap_radar_image_pixels(
            radar_data_old.radar_image,
            radar_data_new.radar_image,
            transformation.affine_matrix_radar_image,
        )
        data = {
            "overlap": [overlap],
            "pixel difference": [pixel_diff_intensity],
            "overlap original": [overlap_original],
            "pixel difference original": [pixel_diff_intensity_original],
        }
        df = pd.DataFrame(data)
        df.to_csv(
            debug_path + "sift_results_" + str(f"{frame_new:03}") + ".csv", index=False
        )
        affine = {
            "affine matrix radar image": [transformation.affine_matrix_radar_image],
            "affine matrix sensor coordinate": [transformation.affine_matrix],
        }
        
        df = pd.DataFrame(affine)
        df.to_csv(
            debug_path + "sift_affine_matrix_" + str(f"{frame_new:03}") + ".csv",
            index=False,
        )
        cv.imwrite(
            debug_path + "match_image_" + str(f"{frame_new:03}") + ".png",
            transformation.matches_img,
        )


#### Test part ###
if __name__ == "__main__":
    path = "/Users/yangxiao/work/python/End-to-end-radar-image-segmentation"
    bscope_folder = path + "/bscope_crop/"
    image_folder = path + "/radar_image_cropped_d1/"
    label_folder = path + "/labels_cropped/"
    data_folder = path + "/radar_data_normalized_d1/"
    path = "debug/"

    frame_end = 4
    for frame_num in range(1, frame_end):
        sift_evaluation(
            path,
            frame_num,
            frame_num + 1,
            SAVE_FEATURE_POINTS_RADAR_IMAGE,
            SAVE_FEATURE_POINTS_BSCOPE,
            SAVE_LABEL_OVERLAP,
            SAVE_RESULTS,
        )



