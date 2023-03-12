import json
import cv2
import numpy as np
import glob

# 1. INITIAL SETUP


chessboard_size = (6, 4)  # calibration chessboard info
size_of_chessboard_squares_mm = 40

right_camera_position = (0, 0, 0)
right_camera_rotation = (0, 0, 0)

right_projection_matrix, _ = cv2.Rodrigues(right_camera_rotation)
right_projection_matrix = np.hstack(
    (right_projection_matrix, np.array(right_camera_position).reshape(-1, 1)))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

obj_p = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
obj_p[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
obj_p = obj_p * size_of_chessboard_squares_mm

left_calibration_images = glob.glob("images_left_separate/*.png")
right_calibration_images = glob.glob("images_right_separate/*.png")
left_stereo_images = glob.glob("left_images/*.png")
right_stereo_images = glob.glob("right_images/*.png")

# 2. CALIBRATING BOTH CAMERAS


obj_points = []
img_points = []
shape = None
# find chessboard on images
for img_name in left_calibration_images:
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        obj_points.append(obj_p)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        shape = gray.shape[::-1]
        img_points.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow("left", img)
        cv2.waitKey(1)
cv2.destroyAllWindows()
obj_points_left, img_points_left, left_shape = obj_points, img_points, shape

obj_points = []
img_points = []
shape = None
# find chessboard on images
for img_name in right_calibration_images:
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        obj_points.append(obj_p)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        shape = gray.shape[::-1]
        img_points.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow("right", img)
        cv2.waitKey(1)
cv2.destroyAllWindows()
obj_points_right, img_points_right, right_shape = obj_points, img_points, shape

# calibrate individual cameras (find distortion coefficients and camera intrinsic matrices)
rpe_l, left_intrinsic_matrix, left_camera_distortion_coefficients, _, _ = cv2.calibrateCamera(
    obj_points_left, img_points_left, left_shape, None, None)
print("Left camera re-projection error", round(rpe_l, 3))

rpe_r, right_intrinsic_matrix, right_camera_distortion_coefficients, _, _ = cv2.calibrateCamera(
    obj_points_right, img_points_right, right_shape, None, None)
print("Left camera re-projection error", round(rpe_r, 3))

# 3. STEREO CALIBRATION TO FIND LEFT CAMERA LOCATION AND ROTATION


obj_points = []
img_points_left = []
img_points_right = []

left_shape = right_shape = None
# find chessboards on images
for nr in range(min(len(left_stereo_images), len(right_stereo_images))):
    img_left = cv2.imread(left_stereo_images[nr])
    img_right = cv2.imread(right_stereo_images[nr])
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # find chessboard corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    if ret_right and ret_left:
        obj_points.append(obj_p)
        corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        left_shape = gray_left.shape[::-1]
        right_shape = gray_right.shape[::-1]
        img_points_left.append(corners2_left)
        img_points_right.append(corners2_right)
        # Draw and display the corners

        cv2.drawChessboardCorners(img_left, chessboard_size, corners2_left, ret_left)
        cv2.imshow('img_left', img_left)
        cv2.drawChessboardCorners(img_right, chessboard_size, corners2_right, ret_right)
        cv2.imshow('img_right', img_right)
        cv2.waitKey(1)
cv2.destroyAllWindows()
obj_points, img_points_left, img_points_right, left_shape, right_shape = \
    obj_points, img_points_left, img_points_right, left_shape, right_shape

# find left camera position and rotation relative to right camera
rpe, _, _, _, _, rot, trans, _, _ = \
    cv2.stereoCalibrate(obj_points, img_points_left, img_points_right, left_intrinsic_matrix,
                        left_camera_distortion_coefficients, right_intrinsic_matrix,
                        right_camera_distortion_coefficients, right_shape)
print("Stereo Calibration re-projection error", round(rpe, 3))
left_camera_rotation, _ = cv2.Rodrigues(rot)
left_camera_position = trans

# set left camera projection matrix
left_projection_matrix = np.hstack((rot, left_camera_position.reshape(-1, 1)))

print("Left camera position (x,y,z): \n", left_camera_position)
print("Left camera rotation vector (degrees): \n", np.degrees(left_camera_rotation))
print("Left camera position and location determined!")

# 4. FIND POINTS ON BOTH IMAGES


board_size = (6, 4)
frame_left = cv2.imread("left_chessboard_image.png")
frame_right = cv2.imread("right_chessboard_image.png")

# convert the input image to a grayscale
gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret_left, corners_left = cv2.findChessboardCorners(gray_left, board_size, None)
ret_right, corners_right = cv2.findChessboardCorners(gray_right, board_size, None)

w = gray_right.shape[1]
h = gray_right.shape[0]

left_points = np.array([np.array([i[0][0], i[0][1]], dtype=np.float32) for i in corners_left], dtype=np.float32)
right_points = np.array([np.array([i[0][0], i[0][1]], dtype=np.float32) for i in corners_right], dtype=np.float32)

# 5. TRIANGULATION


# undistort points
right = cv2.undistortPoints(right_points, right_intrinsic_matrix, right_camera_distortion_coefficients)
left = cv2.undistortPoints(left_points, left_intrinsic_matrix, left_camera_distortion_coefficients)

homogeneous_points = cv2.triangulatePoints(left_projection_matrix, right_projection_matrix, right, left)
points_3d = cv2.convertPointsFromHomogeneous(homogeneous_points.T)

result = [[int(i[0][0]), int(i[0][1]), int(i[0][2])] for i in points_3d]

with open("data.json", "w") as file:
    json.dump(result, file)
