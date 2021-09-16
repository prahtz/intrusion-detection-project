# intrusion-detection-project
A Python project for object detection in a video stream
## Description
This work has the aim of applying the contents explained in the Image Processing and Computer Vision course in a setting of motion detection.
This project is an implementation of a motion detection system, able to detect people and discriminate between real and false object in the scene by applying the following steps:
1. Initialize a background image.
2. For each frame, compute an initial binary image, called change mask, by background subtraction followed by a thresholding operation.
3. Remove noisy blobs from the change mask.
4. Application of binary morphological operators and holes filling.
5. Detect light changes in the frame falsely classified as foreground and remove them from the change mask.
6. Get connected components of the change mask.
7. Update the background image using the frame pixels classified as background in the change mask.
8. Object detection and discrimination between false and true object in the scene.
9. Assign colors and properties to each blob and save the results.
10. (Optional) If there are blobs classified as “false object”, update the background image using their pixels classified as foreground in the change mask.

## Installation and execution
Clone this repoistory and install the depencencies described in the file `requirements.txt`.\
In file `main.py` you can find some examples on how to use the `start_analysis` function of `intrusion_detection.py` file.\
If you want to execute the post-analysis visualization example, go to the `intrusion-detection-project` directory and run 
```
python ./src/main.py
```
