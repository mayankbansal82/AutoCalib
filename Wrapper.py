import cv2
import numpy as np
import glob
from scipy.optimize import least_squares
# import matplotlib.pyplot as plt
import os

# Configuration
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 21.5  # Size of a chessboard square
IMAGE_PATH = "Calibration_Imgs" + "/*.jpg"

# Define the chessboard coordinates in 3D
world_points = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
world_points[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
world_points *= SQUARE_SIZE

# Initialize arrays to hold points from all images
all_corners = []  # 2D image points
all_homographies = []  # Homographies
images = []  # Images themselves
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def makeVi(h, i, j):
    """Constructs elements of the V matrix from a single homography."""
    return np.array([
        h[0][i] * h[0][j], h[0][i] * h[1][j] + h[1][i] * h[0][j], h[1][i] * h[1][j],
        h[0][i] * h[2][j] + h[2][i] * h[0][j], h[1][i] * h[2][j] + h[2][i] * h[1][j], h[2][i] * h[2][j]
    ])

def makeV(hlist):
    """Constructs the V matrix used in camera calibration from a list of homographies."""
    V = []
    for h in hlist:
        V.append(makeVi(h, 0, 1))
        V.append(makeVi(h, 0, 0) - makeVi(h, 1, 1))
    return np.vstack(V)  # Ensure V is a single numpy array for SVD

def calculate_intrinsics(b):
    """Calculate intrinsic camera parameters from the B vector."""
    v0 = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1] ** 2)
    lambda_val = b[5] - (b[3] ** 2 + v0 * (b[1] * b[3] - b[0] * b[4])) / b[0]
    alpha = np.sqrt(lambda_val / b[0])
    beta = np.sqrt(lambda_val * b[0] / (b[0] * b[2] - b[1] ** 2))
    gamma = -b[1] * (alpha ** 2) * (beta / lambda_val)
    u0 = gamma * v0 / beta - b[3] * (alpha ** 2) / lambda_val
    return alpha, beta, gamma, u0, v0

def calculate_A(b):
    """Construct the intrinsic matrix A using the calculated parameters."""
    alpha, beta, gamma, u0, v0 = calculate_intrinsics(b)
    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])
    return A

def calculate_extrinsics(intrinsic_matrix, homographies):
    """Calculate extrinsics from intrinsic matrix and homography matrices."""
    inverse_intrinsic = np.linalg.inv(intrinsic_matrix)
    extrinsics = []

    for H in homographies:
        scale_factor = 1 / np.linalg.norm(np.dot(inverse_intrinsic, H[:, 0]))
        r1 = scale_factor * np.dot(inverse_intrinsic, H[:, 0])
        r2 = scale_factor * np.dot(inverse_intrinsic, H[:, 1])
        t = scale_factor * np.dot(inverse_intrinsic, H[:, 2])
        
        # Reconstruct rotation matrix and translation vector
        R = np.column_stack((r1, r2, np.cross(r1, r2)))
        extrinsic_matrix = np.hstack((R, t.reshape(-1, 1)))
        
        extrinsics.append(extrinsic_matrix)

    return extrinsics

def getx0(A, kc):
    alpha = A[0,0]
    beta = A[1,1]
    gamma = A[0,1]
    u0 = A[0,2]
    v0 = A[1,2]
    k1 = kc[0]
    k2 = kc[1]
    x0 = np.array([alpha, beta, gamma, u0, v0, k1, k2])
    return x0


def lossPlusNewLocations(x, corners, world_points, list_of_Rt):
    error_byImage = []
    alpha, beta, gamma, u0, v0, k1, k2 = x
    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])
    newPoints = []
    number_of_images = 13
    for i, corner in enumerate(corners):
        Rt = list_of_Rt[i]
        e_image = 0
        nan_counter = 0
        points_per_image = world_points.shape[0]
        newPoints_byImage = []
        for j in range(points_per_image):
            point = world_points[j]
            cornerlocation = np.array([corner[j][0], corner[j][1], 1], dtype = 'float').reshape(3,1)
            arr_point = np.array([point[0], point[1], 0, 1])
            projectedPoint = np.dot(Rt, arr_point)
            projectedPoint = projectedPoint / projectedPoint[2]
            x =  projectedPoint[0] 
            y = projectedPoint[1] 
            rad = x*x + y*y
            UVmatr = np.dot(A, projectedPoint)
            UVmatr = UVmatr / UVmatr[2]
            u, v = UVmatr[0], UVmatr[1]
            u1 = float(u + (u-u0)*(k1 * rad + k2 * (rad * rad)))
            v1 = float(v + (v-v0) * (k1 * rad + k2 * (rad * rad)))
            projectedLocation = np.array([u1, v1, 1])
            projectedLocation = projectedLocation.reshape(3, 1)
            diff = projectedLocation - cornerlocation
            squaredDiff = diff * diff
            squaredDist = np.sum(squaredDiff)
            error = np.sqrt(squaredDist)
            if np.isnan(error):
              nan_counter += 1
              continue
            e_image = e_image + error
            newPoints_byImage.append([u1, v1])
        e_image_avg = e_image/((world_points.shape[0] - nan_counter)* number_of_images) 
        error_byImage.append(e_image_avg)
        newPoints.append(newPoints_byImage)
    error_byImage = np.array(error_byImage)

    avg_error = error_byImage.sum() / (world_points.shape[0] * world_points.shape[1])
    return avg_error, newPoints    

def MLE_loss(x, corners, world_points, list_of_Rt):
    error_byImage = []
    alpha, beta, gamma, u0, v0, k1, k2 = x
    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])
    number_of_images = 13
    for i, corner in enumerate(corners):
        Rt = list_of_Rt[i]
        # image_new_points = []
        e_image = 0
        nan_counter = 0
        points_per_image = world_points.shape[0]
        for j in range(points_per_image):
            point = world_points[j]
            cornerlocation = np.array([corner[j][0], corner[j][1], 1], dtype = 'float').reshape(3,1)
            arr_point = np.array([point[0], point[1], 0, 1])
            projectedPoint = np.dot(Rt, arr_point)
            projectedPoint = projectedPoint / projectedPoint[2]
            x =  projectedPoint[0] 
            y = projectedPoint[1] 
            rad = x*x + y*y
            UVmatr = np.dot(A, projectedPoint)
            UVmatr = UVmatr / UVmatr[2]
            u, v = UVmatr[0], UVmatr[1]
            u1 = float(u + (u-u0)*(k1 * rad + k2 * (rad * rad)))
            v1 = float(v + (v-v0) * (k1 * rad + k2 * (rad * rad)))
            projectedLocation = np.array([u1, v1, 1])
            projectedLocation = projectedLocation.reshape(3, 1)
            diff = projectedLocation - cornerlocation
            squaredDiff = diff * diff
            squaredDist = np.sum(squaredDiff)
            error = np.sqrt(squaredDist)
            if np.isnan(error):
              nan_counter += 1
              continue
            e_image = e_image + error
        e_image_avg = e_image/(world_points.shape[0] - nan_counter) * number_of_images
        error_byImage.append(e_image_avg)
    error_byImage = np.array(error_byImage)
    return error_byImage 

def x0toAk(x0):
    alpha, beta, gamma, u0, v0, k1, k2 = x0
    A = np.array([[alpha, gamma, u0],
              [0, beta, v0],
              [0, 0, 1]])
    A = A.reshape(3, 3)
    k = np.array([k1, k2]).reshape(2,1)
    return A, k



# Load images and find chessboard corners
for image_path in glob.glob(IMAGE_PATH):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_image, CHESSBOARD_SIZE, None)

    if ret:
        corners_refined = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
        corners_refined = np.squeeze(corners_refined)
        all_corners.append(corners_refined)
        H, _ = cv2.findHomography(world_points[:, :2], corners_refined)
        all_homographies.append(H)
        images.append(image)

# Calibrate the camera based on the accumulated points and homographies
V = makeV(all_homographies)

_, _, Vt = np.linalg.svd(V)
Vt = np.transpose(Vt)
b = Vt[:,-1]
A = calculate_A(b)

Rt_list = calculate_extrinsics(A, all_homographies)

# Initial guess for distortion coefficients
initial_distortion = (0, 0)

x0 = getx0(A, initial_distortion) #Initialize optimization parameters

# Find error before optimization
loss0, locations0 = lossPlusNewLocations(x0, all_corners, world_points, Rt_list)
print('\nError before optimization: ', loss0)

# Start optimization
sol0 = least_squares(fun=MLE_loss, x0=x0, method="lm", args=[all_corners, world_points,  Rt_list], max_nfev=100)
x1 = sol0.x
A1, K1K2 = x0toAk(x1)
# print(K1K2)
# Rt_list1 = calculate_extrinsics(A1, all_homographies)
loss1, locations1 = lossPlusNewLocations(x1, all_corners, world_points, Rt_list)
print('\nError after optimization: ', loss1, '\n')

K12 = np.array([K1K2.flatten()[0], K1K2.flatten()[1], 0, 0] , np.float32)
print("Final A: ", A1)
print("\nFinal k1, k2: ", K12[0], K12[1])
print("alpha: ", A1[0][0])
print("beta: ", A1[1][1])
print("gamma: ", A1[0][1])
print("u0: ", A1[0][2])
print("v0: ", A1[1][2])

for i, points in enumerate(locations1):
    originalImg = images[i]  # Original image before undistortion
    updatedImg = cv2.undistort(originalImg, A1, K12)
    for point in points:
        x, y = point
        updatedImg = cv2.circle(updatedImg, (int(x), int(y)), 5, (0, 0, 255), 3)

    # Concatenate original and updated images horizontally
    concatenatedImg = np.hstack((originalImg, updatedImg))

    result_images_folder_name = "Result/" 
    if not os.path.exists(result_images_folder_name):
        os.makedirs(result_images_folder_name)
    filelocation = os.path.join(result_images_folder_name, f"combined_{i}.png")
    cv2.imwrite(filelocation, concatenatedImg)
