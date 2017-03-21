# Project Five

## Objectives
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Classification of Cars
For extracting features from a given image I used color transformations as well as histogram of colors. I concatenated these with the HOG features. I tried several variables and ended up using these :

* colorspace='YCrCb'
* orient=9
* pix_per_cell=8
* cell_per_block=2
* hog_channel='ALL'
* spatial_size=(16, 16)
* hist_bins=16
* hist_range=(0, 256)

Next I trained a linear support vector machine on these features (n=6108) and ended up with an test accuracy of 0.9893.

## Sliding window
I used the code from the lecture slides to implement a sliding window approach which classifies each window as either car or non-car. After we found these bounding boxes (see output_images folder for examples) a heatmap was produced by adding 1 to all areas contained in the bounding boxes.
After we obtained a thresholded heatmap we could place rectangles around the cars, as these are the places where the heatmap should have the highest values.

## Video Pipeline
I used the same approach as for the projects before where I used the method process_frame (see continous_vehicle_detection.py) to process each frame independently. I used a smoothing of 20 to get rid of false positives as a car keeps on being detected in consecutive frames.

## Discussion
This approach is not the most robust version of vehicle detection in my opinion. The tuning of the parameters introduces a bias towards the training data. Moreover my implementation is not the fastest with around 2.5it/s. In a real car this needs to be implemented faster! 

