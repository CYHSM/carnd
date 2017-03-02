# Project Four

The fourth project of the udacity self-driving car course wanted to improve on our project one with a more advanced lane finding algorithm. The steps of this project were the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

In the following I want to point out my solutions to these steps!

## Camera Calibration
We are trying to correct images in which the camera should have captured a straight line but due to optical distortion it comes out curved. In order to perform this correction we need to calculate the camera matrix where we need object and image points. The object points are just checkerboard coordinates while the image points are the detected checkerboard edges. I used `cv2.calibrateCamera()` and `cv2.undistort()` for performing the camera calibration, see my code in `advanced_lane_finding.py`, Lines 14-48! The result for one example image looks like the following:

Original             |  Undistorted
:-------------------------:|:-------------------------:
![](https://github.com/CYHSM/carnd/blob/master/CarND-Advanced-Lane-Lines/camera_cal/calibration1.jpg?raw=true)  |  ![](https://github.com/CYHSM/carnd/blob/master/CarND-Advanced-Lane-Lines/output_images/undistorted_example1.jpg?raw=true)

## Thresholded Binary

Now that we estimated the camera matrix for correcting the distortions we can apply these to our images of the road and furthermore use color gradients to form a binary image in which our lane lines should stand out. As there are many parameters to tweak I used the excellent `interact` method from the jupyter notebooks to find the best parameters for seperating lane lines with the different thresholds.
[](https://github.com/CYHSM/carnd/blob/master/CarND-Advanced-Lane-Lines/output_images/jupyter_interact.jpg?raw=true)
The result for one of the test images would look like this:
Undistorted             |  Threshed
:-------------------------:|:-------------------------:
![](https://github.com/CYHSM/carnd/blob/master/CarND-Advanced-Lane-Lines/camera_cal/undistorted_example_road1.jpg?raw=true)  |  ![](https://github.com/CYHSM/carnd/blob/master/CarND-Advanced-Lane-Lines/output_images/threshed_example_road1.jpg?raw=true)

## Perspective Transformation
We now transform the perspective to a birds eye view for fitting a polynomial to the lane lines. For this we need source and destination points, where the `src points` describe a polygon in the original image which will be transformed to a polygon in the `dst points`. I hardcoded the values and used :
```python
src = np.float32([[293, 668], [587, 458], [703, 458], [1028, 668]])
dst = np.float32([[310, im_shape[1]], [310, 0],[950, 0], [950, im_shape[1]]])
```
After perspective transformation the images will loke like this:
Threshed             |  Perspective Transformed
:-------------------------:|:-------------------------:
![](https://github.com/CYHSM/carnd/blob/master/CarND-Advanced-Lane-Lines/camera_cal/threshed_example_road1.jpg?raw=true)  |  ![](https://github.com/CYHSM/carnd/blob/master/CarND-Advanced-Lane-Lines/output_images/transformed_example_road1.jpg?raw=true)

## Fitting a polynomial to the lane
For fitting a polynomial to the perspective transformed and thresholded image we use a sliding histogram approach, where our maximum peaks in the bottom half of the image correspond to the start of the lane. We then subsequently search for the line with the same approach and finally fit a polynomial, which will yield following result:
![](https://github.com/CYHSM/carnd/blob/master/CarND-Advanced-Lane-Lines/camera_cal/fitted_example_road1.jpg?raw=true)

## Video Production
We now can use this do find lane lines in videos by just performing these procedures on each frame subsequently.
My resulting video can be found [here]()
