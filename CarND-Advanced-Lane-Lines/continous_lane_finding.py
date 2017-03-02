import advanced_lane_finding as alf

import numpy as np
from moviepy.editor import VideoFileClip

import cv2


def set_globals():
    global M_g, Minv_g, mtx_g, dist_g, left_fit_g, right_fit_g, curvature_g
    ret, mtx, dist, rvecs, tvecs = alf.load_camera_calibration('./camera_cal')
    M, Minv = alf.load_perspective_matrices()
    left_fit_g, right_fit_g, curvature_g = [], [], []

    M_g, Minv_g, mtx_g, dist_g = M, Minv, mtx, dist


def process_video(video_path, file_out):
    """
    Finds lane lines for given video

    Inputs:
    - video_path : Path to video. o really.-
    """
    set_globals()
    output = file_out
    clip1 = VideoFileClip(video_path)
    # NOTE: this function expects color images!!
    clip = clip1.fl_image(process_frame)
    clip.write_videofile(output, audio=False)


def process_frame(img):
    """
    Processes one frame of an image

    Inputs:
    - img : Frame as color image
    """
    # Undistortion
    undistorted = alf.correct_distortion(img, mtx=mtx_g, dist=dist_g)
    # Sobel & Color thresholds
    opts = {'sobel_kernel': 13, 'stx_min': 20,
            'stx_max': 255, 'sty_min': 20, 'sty_max': 255,
            'mt_min': 20, 'mt_max': 100, 'dt_min': -np.pi / 2, 'dt_max': 1.33,
            'ct_min': 100, 'ct_max': 255}
    binary = alf.combine_sobel_thresholds(undistorted, opts)
    # Perspective transform
    perspective_transformed = alf.transform_to_top_view(binary, M=M_g)
    # Fit polynomial
    left_fit, right_fit, fits = alf.fit_polynomial(
        perspective_transformed, plotit=False, nwindows=15)
    # Smooth fits
    left_fit, right_fit = smooth_fits(left_fit, right_fit)
    # Calculate curvature
    curvature_in_m = alf.calculate_curvature(
        perspective_transformed, left_fit, right_fit, fits)
    # Warp back to road
    result = alf.warp_perspective_back(
        perspective_transformed, img, left_fit, right_fit, fits, Minv=Minv_g)
    # Add curvature to image
    curvature_in_m = np.mean(curvature_in_m)
    curvature_in_m = smooth_curvature(curvature_in_m)
    curvature_text = 'Curvature : {:.2f}'.format(curvature_in_m)
    cv2.putText(result, curvature_text, (200, 100), 0, 1.2, (255, 255, 0), 2)

    return result


def smooth_curvature(curvature, n=50):
    """
    Smoothes the curvature over n frames
    """
    curvature_g.append(curvature)
    curvature_np = np.array(curvature_g)

    if len(curvature_g) > n:
        curvature = np.mean(curvature_np[-n:])

    return curvature

def smooth_fits(left_fit, right_fit, n=20):
    """
    Smoothes the polynomial fits

    Inputs:
    - left, right fit : Polynomial fit of current frame
    """
    left_fit_g.append(left_fit)
    right_fit_g.append(right_fit)

    left_fit_np = np.array(left_fit_g)
    right_fit_np = np.array(right_fit_g)

    if len(left_fit_g) > n:
        left_fit = np.mean(left_fit_np[-n:, :], axis=0)
    if len(right_fit_g) > n:
        right_fit = np.mean(right_fit_np[-n:, :], axis=0)
    return left_fit, right_fit
