
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_test.png "Undistorted"
[image2]: ./output_images/undistort_lane_image.png "Road Undistorted"
[image3]: ./output_images/combined_binary.png "Combined Binary(R +S + gradx) "
[image4]: ./output_images/warped_straight_lines.png "Warped Test image"
[image5]: ./output_images/warp_histogram.png "Fit Visual"
[image6]: ./output_images/sliding_window.png "Sliding Window"
[image7]: ./output_images/pipeline_output.png "Pipeline Output"
[video1]: ./project_video_output.mp4 "Lane annotated Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for the camera calibration step is contained in the code cells [1 - 5] of the IPython notebook located in "./Advanced_Lane_Lines.ipynb".  

I start the calibration process by trying to map image points to real world object points on a set of calibration test images (see cell 2). These are chess board images taken at various perspectives. The "object points", (x, y, z) coordinates of the chessboard corners in the real world are prepared by assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. For example if we have a chess board image with dimensions 9 x 6, then we create an object point 3d array (9x6x1) for each corner of the chess board. The first corner will have the co-ordinate value (0,0,0), the next corner will be (1,0,0) and so on until the last corner cordinate (8,5,0). 

Thus, `objp` is just an array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection using the `cv2.findChessboardCorners()` function. [see cell 3 in notebook]   

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the result below: [*code in notebook cells 4 & 5* ] 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
After calibrating the camera, I applied the calibration matrix and distortion coefficients to a test image using the `cv2.undistort()` function [*code in notebook cell 6*]. The result is shown below

![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
The next step is to take the undistorted image and use color and gradient thresholding to create a binary thresholded image that would bring out the lane lines. 
First we define the following functions [*code in notebook cell 7*]
- grayscale() : to convert the image to grayscle
- abs_sobel_thresh : x & y gradient thresholding using the Sobel operator
- mag_thresh : gradient magnitude thresholding
- dir_thresh : gradient direction threshoding
- color_thresh: common method to do a color thresholding for a channel.

In [*cell 9 of notebook], we apply the different gradient thresholding methods on the undistorted image and show the results of each application. The x Sobel gradient does the best in bringing out the lane lines
In [*cell 10 of notebook] we convert the undistorted image to HLS color space and show the results of S channel thresholding and H channel thresholding. S brings out the yellow line even under different shade conditions. Also did R channel thresholding in the RGB space which was doing a good job of bringing out the white lane lines. Finally chose R & S threshold along with x gradient thresholding to build a combined binary image that brings out the lane lines nicely. See image below.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes 2 functions called `warp_image()` and `unwarp_image` [*see last 2 methods in notebook cell 7*).  The `warp_image()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dest`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32([np.array([[width * 0.45, ht * 0.64],[width*0.55, ht*0.64],[width * 0.88,ht],[width * 0.15,ht]])])
dest = np.float32([np.array([[width * 0.25,0],[width * 0.75,0],[width * 0.75, ht],[width * 0.25, ht]])])

```
[Source code in notebook cell 13]

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 576, 460      | 320, 0        | 
| 704, 460      | 960, 0        |
| 1126,720      | 960, 720      |
| 192, 720      | 320, 720      |

The code to do the warping with source and destination points is defined in the `warp_image()` [* in notebook cell 7 *].  I verified that my perspective transform was working as expected by drawing the `src` and `dest` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Now that we have a warped binary image of the lane lines, we proceed next to finding the lane lines in the image and fitting a 2nd degree polynomial to the line. [* source code for this section is in notebook cells 14 - 18]

First we start by taking a histogram of the lower half of the warped binary image. The histogram is built by taking a sum along the axis. This will produce peaks where there are pixels with ones(lines). Find 2 x points on the left and right half of the images where the histogram has the max value. This will give the starting leftx_base and rightx_base points for each of the lanes. [*code in `find_lane_x_bases()` function in notebook cell 14*]

![alt text][image5]

Next from the 2 x base points, we run two sliding window operations that would move up the image trying to find the lane lines. For each lane we run 9 windows up the y axis. The window height will be image_height/num_windows. The width of the window is built with a margin parameter (100) which determines how many pixels to the left and right of the base point of the lane. So window width would be 201 pixels wide. At each window step we take the mean of the nonzero pixels in the window to find the new x_base for teh next window ( this makes sure that the the windoe follows curved lines) 
[see code in `sliding_window()` method in cell 14*] 

![alt text][image6]

Now that we have found the lane line pts with sliding window we can fit 2nd degree polynomials to the left lane and right lane points [See code in `find_lane_lines()` method in cell 18]. The sliding window operation is a costly operation. We can minimize this cost by making use of the previous polynomial fits to compute new fits for subsequent images. To store the previous pts, fit coeficients, radius of curvature for each image clip, we define a class called `Line()` [* code in cell 17*]. I also added update method in the Line class to keep track of n last fit points and n last fit coefficients. In each update we compute the average of the n fit points and n fit coefficients to determine the best fit points and polynomial fit coefficients. This will allow us to smooth the lane changes between frames. 

Also added a `smooth_fit()` method [*see method code in cell 18*] to update new value with weighted values of previous fits - 20% new value + 80% last fit value. This made the updates even more smoother without jumps.Finally added methods to check if the lane lines are parallel(code in cell 18) and only update the line objects if they are. 


####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

###### Radius of Curvature
I added a function `radius_of_curvature_in_metres()` [*code in cell 18*] to compute the radius of curvature in real world dimensions. In this method, I first compute the scale of x & y in metres_per_pixel as follows

```
    ym_per_pix = 30.0/720  
    xm_per_pix = 3.7 / 700 # the pixel width in between the lanes is 700
    
```
As per the notes in the lecture, the lanes in the camera images are 30 metres long. So divided by the height of the image 720 we get metres per pixel for y values. For the lane width, U.S. regulations require a minimum lane width of 12 feet or 3.7 meters. The distance between the lanes in pixel values is around 700. Using this we compute the metres per pixel for x values.

For the actual radius of curvature calculation we use the formula
<code>
        rad = (( 1+ y'(x)**2)**1.5) / (abs(y''(x))) 

        The equation for lane's 2nd degree polynomial fit is
        
        x = A*y**2 + B*y +C

        y'(x)  = 2*A*y + B
        y''(x) = 2*A
</code>

Using the above we compute the radius of curvature for the lanes in metres. 

###### Vehicle offset from center
The code to calculate vehicle offset from center is in the `lane_offset_from_center()` method. [* code in notebook cell 19]. 
We do this by calculating the midpoint of the image and the midpoint of the lanes, then taking their difference and finally converting it into metres using the 'x metres per pixel' scale.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this in the `pipeline()` method [in cell 20 of the notebook]. In the method, I do all the steps from undistorting the image, to computing gradient and color thresholds, to warping the image and finding lanes. Once we have the lanes, we draw the lanes on a blank image, then unwarp it with the unwarp function [* defined in notebook cell 7*]. Then we lay this image over the undistorted using `cv2.add_weighted()` function. Finally we anotate the image with radius of curvature values for each lanes and the offset from center in metres. The image below shows how a test image run through the pipeline will result in the annotated image

![alt text][image7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the problems that I faced early on was that the S threshold was including lane shadow of trees that go across the lanes. So it looked like the lan was turning hard left. but this was only for one of the image frames in the clip. I was able to use smoothing and averaging over frames to resolve this. 

The pipeline works very well with the project_video. However in the challenge clip, the dividers shadow falls very close to teh left lane which is is picked up by the Sobelx gradient. So this line confuses the sliding window logic when the lane curves and the shadow seem to spill over the lane. I could use only S & R thresholding which does a reasonable good job of finding the lanes to resolve this.

The system is still not very robust and we have to play around with different color spaces and thresholding for different challenging road conditions. Should look into more advanced lane sensing algorithms that probably uses high resolution LIDAR.

  

