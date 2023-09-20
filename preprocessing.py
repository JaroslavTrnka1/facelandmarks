import os
import cv2
import numpy as np
from tqdm import tqdm
from landmarks_utils import readtps, MediaPipe_model, LBF_model
from cropping import crop_face_only
from config import *


def get_sample_groups():
    sample_groups = []
    for root, dirs, files in os.walk('./AI_Morphometrics', topdown=True):
        if not dirs:
            sample_groups.append(root)
    return sample_groups


def prepare_training_landmarks(both_models = BOTH_MODELS):
    if both_models:
        x = np.empty((0, 546 * 2))
    else:
        x = np.empty((0, 478 * 2))
    y_true = np.empty((0, 144))
    
    face_detail_coordinates = np.empty((0, 4))
    groups = get_sample_groups()
    path_list = []
    
    for group in tqdm(groups):

        for file in os.listdir(group):
            if '.TPS' in file or '.tps' in file:
                tps = readtps(group + '/' + file, group)

                for idx in range(len(tps['im'])):
                    true_landmarks = tps['coords'][:, :, idx]
                                      
                    try:
                        
                        img_path = group + '/' + tps['im'][idx]
                        image = cv2.imread(img_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # scaling from pixel to (0,1)
                        true_landmarks = np.divide(true_landmarks, (image.shape[1], image.shape[0]))
                        # As the source true landmarks have y-origin on the bottom of picture
                        # (unlike MediaPipe model)
                        # we have to flip their y-axis
                        true_landmarks[:,1] = 1 - true_landmarks[:,1]

                        # We crop only face detail for a better precision
                        # and accomodate landmark coordinates to cropped subimage
                        # as well as to float (0,1) scale
                        # face_detail = crop_face_only(image)
                        
                        # if face_detail:
                        #     subimage, xmin, ymin, xmax, ymax = face_detail
                        #     true_landmarks = np.subtract(true_landmarks, (xmin, image.shape[0] - ymax))
                        #     true_landmarks = np.divide(true_landmarks, (subimage.shape[1], subimage.shape[0]))



                        # Both models use float(0,1) for x and y axis
                        # cv2 returns upper left axis origin by default - in this form the image is processed by both models
                    
                        if both_models:
                            input_landmarks = np.concatenate((LBF_model(image), MediaPipe_model(image)), axis = 0)
                        else:
                            input_landmarks = MediaPipe_model(image)
                        
                        input_landmarks = input_landmarks.reshape(1,-1)
                        x = np.concatenate((x, input_landmarks), axis = 0)

                        true_landmarks = true_landmarks.reshape(1,-1)
                        y_true = np.concatenate((y_true, true_landmarks), axis = 0)
                        
                        path_list.append(img_path)
                        
                    except:
                        # Wrong image wasn't accepted into preprocessed data
                        print(f"Preprocessing went wrong in image: {img_path}.")

    return x, y_true, path_list



def save_preprocessed_data(x_inp, y_inp, path_list):
    if not  os.path.exists('preprocessed_data'):
        os.mkdir('preprocessed_data')
    np.savez('preprocessed_data/preprocessed_inputs', x_inp = x_inp, y_inp = y_inp)
    
    with open("preprocessed_data/path_list.txt", "w") as pl:
        for path in path_list:
            pl.write(str(path) +"\n")


if not os.path.isfile("preprocessed_data/path_list.txt"):  
    x_inp, y_inp, path_list = prepare_training_landmarks()   
    try:
        print(x_inp.shape)
        print(y_inp.shape)
        print(len(path_list)) 
    except Exception as e:
        print(e)
        
    save_preprocessed_data(x_inp, y_inp, path_list)