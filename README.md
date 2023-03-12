# opencv-triangulation-example

This Python code demonstrates how to perform triangulation with OpenCV. In this specific case, it finds the 3d coordinates (with origin of coordinate system at the left camera) of the inner corners of a chessboard that is seen from both cameras.

  
  
I put it up on GitHub in the hope that it can help other beginners like myself who are working on similar projects and need an example of triangulation with OpenCV. If I had had access to code like this, it would have probably saved me like 40 hours of debugging and confusion.  
  
Please note that while this code works for me, I can't say for sure that everything is 100% correct. There may be errors or issues that I haven't encountered. However, I hope that by sharing this code, other beginners can use it as a starting point and improve upon it to suit their needs.
  
This code could be formatted much better with functions/classes etc., but I left it as is in order to give a clear overview of the order of steps taken.
  
The code expects the folder to be organized as:
```
- root
  - images_left_separate
    - img0.png
    - img1.png
    - ..
  - images_right_separate
    - img0.png
    - img1.png
    - ..
  - left_images 
    - img0.png
    - img1.png
    - ..
  - right_images
    - img0.png
    - img1.png
    - ..
  - triangulation_example.py
  - left_chessboard_image.png
  - right_chessboard_image.png
```
  
For camera calibration around 30-40 images is optimal with chessboard seen from different angles and distances.
For stereo calibration a couple of images is good enough.
  
TLDR main opencv functions used: 
- `calibrateCamera` 2x
- `stereoCalibrate`
- `undistortPoints`
- `triangulatePoints`
- `convertPointsFromHomogeneous`

