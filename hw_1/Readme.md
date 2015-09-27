# First Homework

## Problem 1

### Goal

Implement an object recognition system.

### Solution

The system was build using 110 threshold for binorization. The comparison of objects were made using
the roundness criteria only with threshold of 0.04.

The results can be seen on the following picture. On the left is the original file with the images of objects
to find and on the right are the images where the objects were detected. The pairs of images show the matched objects
their orientation on both images and the center of mass of the object.

To run code just `python task_1.py` while bein in the hw_1 directory.

![Alt text](task_1_results.png?raw=true "Optional Title")
![Alt text](task_1_results_2.png?raw=true "Optional Title")

## Problem 2

### Goal

Implement Hough transform for searching lines in images.

### Solution

For the last part where it was needed to find the begining of the line segment and the end a simple algo was used:
first the equation of the line was taken and the respective coordinates of thresholded image with magnitudes of
gradient was used. The algorithm walks along the line and when it finds non-zero element it marks this part as a start
of line segment. Then it continues to walk until the zero element is reached. After this the line segment is saved.

The following constants was used:

```
AMOUNT_OF_BINS_THETA = 50
AMOUNT_OF_BINS_RO = 100
```

`500` the magnitude threshold.
`900` hough threshold.
The voting was simply implemented by doing counts.

![Alt text](task_2_results.png?raw=true "Optional Title")
![Alt text](task_2_results_2.png?raw=true "Optional Title")
![Alt text](task_2_results_3.png?raw=true "Optional Title")
