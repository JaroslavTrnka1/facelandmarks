import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
import urllib.request as urlreq
import torch
from config import *


def readtps(input, path):
    """
    Function to read a .TPS file
    Args:
        input (str): path to the .TPS file
    Returns:
        lm (str list): info extracted from 'LM=' field
        im (str list): info extracted from 'IMAGE=' field
        id (str list): info extracted from 'ID=' filed
        coords: returns a 3D numpy array if all the individuals have same
                number of landmarks, otherwise returns a list containing 2d
                matrices of landmarks
    """

    # open the file
    #tps_file = io.open(input, mode="r", encoding="latin-1")
    tps_file = open(input, 'r')  # 'r' = read
    tps = tps_file.read().splitlines()  # read as lines and split by new lines
    tps_file.close()

    # initiate lists to take fields of "LM=","IMAGE=", "ID=" and the coords
    lm, im, ID, coords_array, skipped_imgs = [], [], [], [], []
    skipping = False

    # looping thru the lines
    for i, ln in enumerate(tps):

        # Each individual starts with "LM="
        if ln.startswith("LM"):
            # number of landmarks of this ind
            lm_num = int(ln.split('=')[1])

            if lm_num == 72:

                # fill the info to the list for all inds
                lm.append(lm_num)
                # initiate a list to take 2d coordinates
                coords_mat = []

                # fill the coords list by reading next lm_num of lines
                for j in range(i + 1, i + 1 + lm_num):
                    coords_mat.append(tps[j].replace(',', '.').split(' '))  # split lines into values

                # change the list into a numpy matrix storing float vals
                coords_mat = np.array(coords_mat, dtype=float)
                # fill the ind 2d matrix into the 3D coords array of all inds
                coords_array.append(coords_mat)

            else:
                print(f'Skipped image in: {path}.')
                skipping = True

        # Get info of IMAGE= and ID= fields
        if ln.startswith("IMAGE"):
            if not skipping:
                im.append(ln.split('=')[1])

        if ln.startswith("ID"):
            if not skipping:
                ID.append(ln.split('=')[1])
            skipping = False

    # check if all inds contains same number of landmarks
    all_lm_same = all(x == lm[0] for x in lm)
    # if all same change the list into a 3d numpy array
    if all_lm_same:
        coords_array = np.dstack(coords_array)
    else:
        print('Images with different number of landmarks!')

    # return results in dictionary form
    return {'lm': lm, 'im': im, 'id': ID, 'coords': coords_array}


def LBF_model(image, prepared = False):
    if not prepared:
        # save face detection algorithm's name as haarcascade
        haarcascade = "haarcascade_frontalface_alt2.xml"

        # save facial landmark detection model's name as LBFmodel
        LBFmodel = "lbfmodel.yaml"

        if (haarcascade not in os.listdir(os.curdir)):
            haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
            urlreq.urlretrieve(haarcascade_url, haarcascade)

        if (LBFmodel not in os.listdir(os.curdir)):
            LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
            urlreq.urlretrieve(LBFmodel_url, LBFmodel)

        # create an instance of the Face Detection Cascade Classifier
        detector = cv2.CascadeClassifier(haarcascade)
        # create an instance of the Facial landmark Detector with the model
        landmark_detector = cv2.face.createFacemarkLBF()
        landmark_detector.loadModel(LBFmodel)

    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(image_gray)

    # Detect landmarks on "image_gray"
    _, landmarks = landmark_detector.fit(image_gray, faces)
    landmarks = landmarks[0][0,:,:]

    # Returns coordinates as float from 0 to 1
    landmarks = np.divide(landmarks, (image.shape[1], image.shape[0]))

    return landmarks


def MediaPipe_model(image):
    landmarks = np.empty((478, 2))
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,    # Zpřesnění kolem očí a pusy!!
        min_detection_confidence=0.5) as face_mesh:

        # To improve performance
        image.flags.writeable = False

        # Detect the face landmarks
        results = face_mesh.process(image)

        # To improve performance
        #image.flags.writeable = True

    if len(results.multi_face_landmarks) == 1:
        for i, l_mark in enumerate(results.multi_face_landmarks[0].landmark):
                landmarks[i,:] = l_mark.x, l_mark.y

    else:
        print("Single face not detected! Only one face per picture for landmarking.")

    return landmarks

def display_landmarks(landmarks, img, pixel_scale = False, origin = None, errors = None):

    x_coor = landmarks[:,0]
    y_coor = landmarks[:,1]

    if not pixel_scale:
        x_coor = np.multiply(x_coor, img.shape[1])
        y_coor = np.multiply(y_coor, img.shape[0])

    if not origin == 'upper_left':
        y_coor = img.shape[0] - y_coor

    fig, ax = plt.subplots(1, figsize = (10,10))
    ax.imshow(img)
    ax.axis('off')

    for i, (x, y) in enumerate(zip(x_coor, y_coor)):
        circ1 = Circle((x,y), 2, color='white', fill = False)
        ax.add_patch(circ1)

        if errors is None:
            pass
        else:
            err_x = errors[i,0] * img.shape[1]
            err_y = errors[i,1] * img.shape[0]
            circ2 = Ellipse((x,y), err_x, err_y, color = 'blue', fill = False)
            ax.add_patch(circ2)

    plt.show()


def get_relative_positions(landmarks):
    margin_coef = torch.tensor([1.7, 1.85]).to(DEVICE)
    
    # Calculate the mean absolute position of landmarks
    centroid = torch.mean(landmarks, axis=-2, keepdim=True)

    # Calculate relative positions by subtracting the mean
    relative_landmarks = torch.subtract(landmarks, centroid)

    # Calculate the maximum absolute distance from the mean position
    max_distance = torch.mul(torch.max(torch.abs(relative_landmarks), axis=-2, keepdim = True)[0], margin_coef)

    # Normalize relative positions to the range [0, 1] by dividing by the maximum absolute distance
    relative_landmarks = torch.div(torch.add(relative_landmarks, max_distance), (2 * max_distance))

    # Calculate a measure of size (maximum absolute distance)
    size_measure = max_distance

    return relative_landmarks, centroid, size_measure

def fit_to_relative_centroid(landmarks, centroid, size_measure):
    
    relative_landmarks = torch.subtract(landmarks, centroid)
    relative_landmarks = torch.div(torch.add(relative_landmarks, size_measure), (2 * size_measure))
    
    return relative_landmarks

def get_absolute_positions(relative_landmarks, centroid, size_measure):

    # Scale the relative positions by the size_measure and add the mean position
    absolute_landmarks = torch.sub(2 * torch.mul(relative_landmarks, size_measure), torch.add(size_measure, -1 * centroid))

    return absolute_landmarks

def get_face_angle(landmarks, image_shape):
    forehead = (landmarks[151, 0] * image_shape[1], landmarks[151, 1] * image_shape[0])
    chin = (landmarks[152, 0] * image_shape[1], landmarks[152, 1] * image_shape[0])
    angle_rad = np.arctan2(chin[0].item() - forehead[0].item() , chin[1].item() - forehead[1].item()) * -1
    angle_deg = np.degrees(angle_rad)
    return angle_deg.item()

def rotate_landmarks(angle_deg, landmarks, image_shape):
    center = torch.tensor([image_shape[1]//2, image_shape[0]//2])
    pixel_landmarks = torch.mul(landmarks, torch.tensor([image_shape[1], image_shape[0]]))
    rotation_matrix = cv2.getRotationMatrix2D((center[0].item(), center[1].item()), angle_deg, 1.0)

    centered_landmarks = pixel_landmarks - center
    centered_landmarks = torch.matmul(centered_landmarks, torch.tensor(rotation_matrix[:,:2], dtype=torch.float32).T)

    rotated_landmarks = centered_landmarks + center
    rotated_landmarks = torch.div(rotated_landmarks, torch.tensor([image_shape[1], image_shape[0]]))
    return rotated_landmarks

def display_parent_landmarks(projection_mask, img_idx, landmark_idx):
    preprocessed_inputs = np.load('preprocessed_data/preprocessed_inputs.npz')
    inputs = preprocessed_inputs['x_inp']
    targets = preprocessed_inputs['y_inp']
    path_list = []
    with open("preprocessed_data/path_list.txt", "r") as pl:
        for path in pl:
            path_list.append(path.strip())
            
    idx = img_idx
    img = cv2.imread(path_list[idx])

    inp = inputs[idx,:]
    targ = targets[idx,:].reshape(-1,2)

    landmark_idx = landmark_idx
    target_landmark = targ[landmark_idx:landmark_idx+1, :]

    mask = projection_mask[2 * landmark_idx,:]
    input_landmarks = inp[mask].reshape(-1,2)

    display_landmarks(target_landmark, img, pixel_scale=False, origin='upper_left')
    display_landmarks(input_landmarks, img, pixel_scale=False, origin='upper_left')

# landmarks = torch.tensor([[100, 150], [120, 160], [90, 145], [110, 155]], dtype = torch.float64)
# relative_landmarks, centroid, size_measure = get_relative_positions(landmarks)

# torch.mul(centroid, torch.tensor([2,3], dtype = int)).type(torch.int)

# import torch
# a = torch.randn(3,4,4)
# #torch.flatten(a, start_dim = -2) == a.reshape(3,-1)

# a.unflatten(-1, (-1,2)) == a.reshape(3, 4, 2, 2)

# landmarks = torch.randn(20,50,10,2)
# get_relative_positions(landmarks)[0].shape
# relative_landmarks, centroid, size_measure = get_relative_positions(landmarks)
# get_absolute_positions(relative_landmarks, centroid, size_measure).shape


