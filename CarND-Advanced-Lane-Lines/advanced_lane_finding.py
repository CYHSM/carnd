import glob
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import cv2


# ----------------------------------------------------
# ---------Camera Calibration-------------------------
# ----------------------------------------------------
def camera_calibration(folderpath, ny=6, nx=9):
    """
    Calculates the camera matrix and the distortion coefficients and saves them for later use in the correction.

    Inputs:
    - folderpath : Path to folder with camera calibration images
    """
    cal_images = glob.glob(folderpath + '/calibration*.jpg')
    image_points, object_points = [], []
    # Prepare checkerboard points
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    # Loop over all calibration images and calculate image points
    for img_path in cal_images:
        img = mpimg.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Find chessboard corners (all edges, not just outer ones)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            image_points.append(corners)
            object_points.append(objp)
    # Calculate camera matrix and distortion coefficients
    calibration_data = cv2.calibrateCamera(
        object_points, image_points, gray.shape[::-1], None, None)

    pickle.dump(calibration_data, open(folderpath + '/calibration.p', 'wb'))


def load_camera_calibration(folderpath='./camera_cal'):
    return pickle.load(open(folderpath + '/calibration.p', 'rb'))


def correct_distortion(img, mtx=None, dist=None):
    return cv2.undistort(img, mtx, dist, None, mtx)

# ----------------------------------------------------
# ---------Sobel, Color and Gradient Threshold--------
# ----------------------------------------------------


def color_threshold(img, color_thresh=(0, 255)):
    """
    Calculate threshold on s channel for HLS image

    Inputs:
    - img : Undistorted image
    - color_thresh : Threshold for color channel

    Outputs:
    - Binary mask where color channel is in range of color_thresh
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    s_mask = np.zeros_like(s_channel)
    s_mask[(s_channel >= color_thresh[0]) & (s_channel <= color_thresh[1])] = 1

    return s_mask


def abs_sobel_threshold(img, orient='x', sobel_kernel=3,
                        sobel_thresh=(0, 255)):
    """
    Calculate sobel derivation on given orientation

    Inputs:
    - img : Undistorted image
    - orient : Orientation on which to calculate the derivative (x will enhance vertical lines, y enhances horizontal lines)
    - sobel_kernel : Sobel kernel size
    - sobel_thresh : Threshold for binarizing derivative

    Outputs:
    - Binary mask where derivate is in range of sobel_thresh
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel = np.abs(sobel)
    sobel = np.uint8(255 * (sobel / np.max(sobel)))
    # Binarize derivative
    sobel_mask = np.zeros_like(sobel)
    sobel_mask[(sobel > sobel_thresh[0]) & (sobel <= sobel_thresh[1])] = 1

    return sobel_mask


def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Calculates the magnitude of the sobel derivative and returns threshed values

    Inputs:
    - img : Undistorted image
    - sobel_kernel : Sobel kernel size
    - mag_thresh : Threshold for binarizing magnitude

    Outputs:
    - Binary mask where magnitude is in range of thresh
    """

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0,  ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # The magnitude is the square-root of the squared sum
    sobel_magnitude = np.sqrt(np.power(sobel_x, 2) + np.power(sobel_y, 2))
    sobel_magnitude = np.uint8(
        255 * (sobel_magnitude / np.max(sobel_magnitude)))
    # Binarize magnitude
    sobel_mask = np.zeros_like(sobel_magnitude)
    sobel_mask[(sobel_magnitude > mag_thresh[0]) &
               (sobel_magnitude <= mag_thresh[1])] = 1

    return sobel_mask


def dir_threshold(img, sobel_kernel=3,
                  dir_thresh=(0, np.pi / 2)):
    """
    Calculates the direction of the gradient:

    arctan(sobel_x / sobel_x)

    Resulting image contains a value for the angle of the gradient away from horizontal in radians (-pi/2, pi/2). Orientation of 0 implies a horizontal line while (-pi/2, pi/2) imply vertical lines.

    Inputs:
    - img : Undistorted image
    - sobel_kernel : Sobel kernel size
    - dir_thresh : Threshold for binarizing magnitude

    Outputs:
    - Binary mask where magnitude is in range of dir_thresh
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel_x = np.abs(sobel_x)
    abs_sobel_y = np.abs(sobel_y)
    # Calculate direction of gradient
    sobel_direction = np.arctan2(abs_sobel_y, abs_sobel_x)
    # Binarize direction
    sobel_mask = np.zeros_like(sobel_direction)
    sobel_mask[(sobel_direction > dir_thresh[0]) &
               (sobel_direction <= dir_thresh[1])] = 1

    return sobel_mask


def combine_sobel_thresholds(img, opts):
    """
    Combines the sobel derivative, magnitude and direction:

    Inputs:
    - img : Undistorted image
    - options : A dictionary with the values for transforming the image
        - sobel_kernel : Kernel size
        - stx_min, stx_max : Sobel threshold for x
        - sty_min, sty_max : Sobel threshold for y
        - mt_min, mt_max : Magnitude threshold
        - dt_min, dt_max : Direction threshold

    Outputs:
    - Binary mask where all thresholds are met
    """

    # Calculate sobel gradient, magnitude and direction
    gradx = abs_sobel_threshold(img, orient='x', sobel_kernel=opts['sobel_kernel'], sobel_thresh=(
        opts['stx_min'], opts['stx_max']))
    grady = abs_sobel_threshold(img, orient='y', sobel_kernel=opts['sobel_kernel'], sobel_thresh=(
        opts['sty_min'], opts['sty_max']))
    mag_binary = mag_threshold(img, sobel_kernel=opts['sobel_kernel'],
                               mag_thresh=(opts['mt_min'], opts['mt_max']))
    dir_binary = dir_threshold(
        img, sobel_kernel=opts['sobel_kernel'], dir_thresh=(opts['dt_min'], opts['dt_max']))
    color_binary = color_threshold(
        img, color_thresh=(opts['ct_min'], opts['ct_max']))
    # Combine thresholded images
    combined = np.zeros_like(dir_binary)
    # combined[((gradx == 1) & (grady == 1)) | (
    #     (mag_binary == 1) & (dir_binary == 1))] = 1
    combined[((gradx == 1) & (grady == 1) &
              (mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1)] = 1

    return combined


# ----------------------------------------------------
# ---Perspective Transform and Polynomial Fit---------
# ----------------------------------------------------
def calculate_perspective_matrices(img, src=None, dst=None):
    """
    Performs a perspective transformation on given image

    Inputs:
    - img : Undistorted image

    Outputs:
    - Transformed image
    """
    im_shape = (img.shape[1], img.shape[0])
    if src is None and dst is None:
        # src = np.float32([[200, im_shape[1]], [551, 468], [723, 468], [1118, im_shape[1]]])
        src = np.float32([[293, 668], [587, 458], [703, 458], [1028, 668]])
        dst = np.float32([[310, im_shape[1]], [310, 0],
                          [950, 0], [950, im_shape[1]]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    pickle.dump([M, Minv], open('./persp_transform.p', 'wb'))


def load_perspective_matrices():
    M, Minv = pickle.load(open('./persp_transform.p', 'rb'))
    return M, Minv


def transform_to_top_view(img, M):
    """
    Performs a perspective transformation on given image

    Inputs:
    - img : Undistorted image
    - M : Transformation matrix

    Outputs:
    - Transformed image
    """
    im_shape = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, im_shape)


def fit_polynomial(img, nwindows=9, plotit=False):
    """
    Finds the line with the help of an histogram and fits a second order polynomial. Code taken from course page

    Inputs:
    - img : Perspective transformed and undistorted image

    Outputs:
    - left_fit, right_fit : Coefficients of left and right line
    """
    # Assuming you have created a warped binary image called "binary_warped"
    binary_warped = img
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # # Choose the number of sliding windows
    # nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean
        # position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[
                        0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + \
        right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[
        left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[
        right_lane_inds]] = [0, 0, 255]

    if plotit:
        plt.imshow(out_img / 255)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return left_fit, right_fit, (leftx, lefty, rightx, righty, ploty)


def fit_polynomial_consecutive(img, left_fit, right_fit, nwindows=9, plotit=False):
    """
    Finds the line with the help of an histogram and fits a second order polynomial. Takes already found fits

    Inputs:
    - img : Perspective transformed and undistorted image

    Outputs:
    - left_fit, right_fit : Coefficients of left and right line
    """
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack(
        (binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[
        left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[
        right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array(
        [np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    if plotit:
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return left_fit, right_fit, (leftx, lefty, rightx, righty, ploty)


def calculate_curvature(img, left_fit, right_fit, fits):
    """
    Calculates curvature for given polynomial fits

    Returns:
    - left_curverad, right_curverad : Radius of curvature in meters

    """
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = img.shape[0]
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(fits[1] * ym_per_pix, fits[0] * xm_per_pix, 2)
    right_fit_cr = np.polyfit(fits[3] * ym_per_pix, fits[2] * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[
                     1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[
                      1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad


def warp_perspective_back(warped, undist, left_fit, right_fit, fits, Minv):
    """
    Warps perspective back to original view and draws lane area

    Inputs:
    - img : Undistorted and perspective transformed image
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    left_fitx = left_fit[0] * fits[4]**2 + left_fit[1] * fits[4] + left_fit[2]
    right_fitx = right_fit[0] * fits[4]**2 + right_fit[1] * fits[4] + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, fits[4]]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, fits[4]])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective
    # matrix (Minv)
    newwarp = cv2.warpPerspective(
        color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # plt.imshow(result)
    return result
