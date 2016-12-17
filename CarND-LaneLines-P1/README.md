# Project One
In the first project of the udacity self-driving car course we had to find lane markings in pictures and videos of streets. See [here](https://github.com/CYHSM/carnd/blob/master/CarND-LaneLines-P1/README_Udacity.md) for the official Udacity Readme for the first project.

The detection of the lane marking was basically done in three steps:
1. Use a gaussian kernel to filter the image
  * This helps in getting rid of noisy parts of the image which makes the next steps more reliable
2. Perform canny edge detection
  * This step basically detects the edges in the image with the help of the image gradient and hysteris (see [here](https://en.wikipedia.org/wiki/Canny_edge_detector) for more details)
3. Use hough transformation to find lines from the edges
  * Transforms each point to a line in hough space where the intersection of these lines shows the presence of a line in image space (see [here](https://en.wikipedia.org/wiki/Hough_transform))

For the test images

Test Image             |  Blurred
:-------------------------:|:-------------------------:
![](https://github.com/CYHSM/carnd/blob/master/CarND-LaneLines-P1/test_images/pipeline/original.jpg?raw=true)  |  ![](https://github.com/CYHSM/carnd/blob/master/CarND-LaneLines-P1/test_images/pipeline/blur.jpg?raw=true)

Canny edge detection             |  Hough transformation
:-------------------------:|:-------------------------:
![](https://github.com/CYHSM/carnd/blob/master/CarND-LaneLines-P1/test_images/pipeline/canny.jpg?raw=true)  |  ![](https://github.com/CYHSM/carnd/blob/master/CarND-LaneLines-P1/test_images/pipeline/lines.jpg?raw=true)
