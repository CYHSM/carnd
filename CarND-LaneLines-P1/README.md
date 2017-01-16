# Project One
In the first project of the udacity self-driving car course we had to find lane markings in pictures and videos of streets. Click [here](https://github.com/CYHSM/carnd/blob/master/CarND-LaneLines-P1/README_Udacity.md) for the official Udacity readme regarding this project.

The detection of the lane marking was basically done in three steps:

1. Use a gaussian kernel to filter the image
  * This helps in getting rid of noisy parts of the image which makes the next steps more reliable
2. Perform canny edge detection
  * This step basically detects the edges in the image with the help of the image gradient and hysteris (see [here](https://en.wikipedia.org/wiki/Canny_edge_detector) for more details)
3. Use hough transformation to find lines from the edges
  * Transforms each point to a line in hough space where the intersection of these lines shows the presence of a line in image space (see [here](https://en.wikipedia.org/wiki/Hough_transform))

## Example Image

Test Image             |  Blurred
:-------------------------:|:-------------------------:
![](https://github.com/CYHSM/carnd/blob/master/CarND-LaneLines-P1/test_images/pipeline/original.jpg?raw=true)  |  ![](https://github.com/CYHSM/carnd/blob/master/CarND-LaneLines-P1/test_images/pipeline/blur.jpg?raw=true)

Canny edge detection             |  Hough transformation
:-------------------------:|:-------------------------:
![](https://github.com/CYHSM/carnd/blob/master/CarND-LaneLines-P1/test_images/pipeline/canny.jpg?raw=true)  |  ![](https://github.com/CYHSM/carnd/blob/master/CarND-LaneLines-P1/test_images/pipeline/lines.jpg?raw=true)

## Connecting the lines

One of the more tricky parts of this project was to connect the lines you see in the picture after the hough transformation. I used following procedure:

1. Get the slope of each line and thresh all lines smaller than 0.5
2. Calculate the size of each line and take the average slope of the biggest x lines.
  * I used x=5
3. Get b with `b = y - m * x` and again get average b of the x biggest lines.
4. With this line equation we can extrapolate to the bottom and the top of the lines.

## Performance on Videos
We had to test our algorithm on different videos with increasing difficulty. I used smoothing over the single frames with a window size of 10. Here are the results of this algorithm (see [Jupyter Notebook](https://github.com/CYHSM/carnd/blob/master/CarND-LaneLines-P1/P1.ipynb) for more details). Click on the videos to view in full quality on youtube.

### Video White
[![Video White](https://github.com/CYHSM/carnd/blob/master/CarND-LaneLines-P1/white.gif?raw=true)](https://youtu.be/bajgiQuEc38)
### Video Yellow
[![Video Yellow](https://github.com/CYHSM/carnd/blob/master/CarND-LaneLines-P1/yellow.gif?raw=true)](https://youtu.be/-F1OEQzTvcg)
### Video Challenge
[![Video Challenge](https://github.com/CYHSM/carnd/blob/master/CarND-LaneLines-P1/extra.gif?raw=true)](https://youtu.be/g3G53Fcfb64)
