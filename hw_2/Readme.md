# Second Homework

## Problem 1

### Goal

Implement the panorama stitching algo using different methods.

### Solution

The system was implemented using harris corner detection, simple patch feature descriptor and simple sift descriptor, 
homography and affine transformation estimation and ransac for robust feature matching.

On the most images the matching worked well except the graph_3 image because the change is very big and the descriptors
for the transformed corners are very different.

Affine transformation didn't worked really well in some cases because it doesn't work in case of projective transformations.

![Alt text](bikes_1_3.jpg?raw=true "Optional Title")
![Alt text](bikes_1_2.jpg?raw=true "Optional Title")

![Alt text](graph_1_2_panorama.jpg?raw=true "Optional Title")

![Alt text](leuven_1_2.jpg?raw=true "Optional Title")
![Alt text](leuven_1_3.jpg?raw=true "Optional Title")

![Alt text](wall_1_2.jpg?raw=true "Optional Title")

