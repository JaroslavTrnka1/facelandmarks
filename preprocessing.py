import os
import cv2
import numpy as np
from tqdm import tqdm
from facelandmarks.landmarks_utils import readtps, MediaPipe_model, LBF_model, get_face_angle
from facelandmarks.cropping import crop_face_only
from facelandmarks.config import BOTH_MODELS

def file2lowercase(file_path):
    new_file_path = file_path[:-4]
    new_extension = file_path[-4:].lower()
    new_file_path = new_file_path + new_extension
    os.rename(file_path, new_file_path)

def get_sample_groups(root_dirs = None):
    if root_dirs is None:
        root_dirs = ['./AI_Morphometrics']
    sample_groups = []
    for root_dir in root_dirs:
        for root, dirs, files in os.walk(root_dir, topdown=True):
            if not dirs:
                sample_groups.append(root.replace("\\","/"))
    return sample_groups

def prepare_training_landmarks(both_models = BOTH_MODELS, group = None):
    if both_models:
        x = np.empty((0, 546 * 2))
    else:
        x = np.empty((0, 478 * 2))
    y_true = np.empty((0, 144))
    
    # face_detail_coordinates = np.empty((0, 4))

    path_list = []
    angles = np.empty(0)
    crops = np.empty((0,4))

    for file in os.listdir(group):
        if file[-4:] in ['.TPS', '.tps']:
            tps = readtps(group + '/' + file, group)

            for idx in range(len(tps['im'])):
                true_landmarks = tps['coords'][:, :, idx]
                                    
                try:
                    img_file = tps['im'][idx]
                    filename, extension = img_file.split('.')
                    if extension in ['JPG', 'BMP']:
                        if os.path.exists(f'{group}/{img_file}'):
                            file2lowercase(f'{group}/{img_file}')
                        img_path = f'{group}/{filename}.{extension.lower()}'
                    else:
                        img_path = f'{group}/{filename}.{extension}'

                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # scaling from pixel to (0,1)
                    # true_landmarks = np.divide(true_landmarks, (image.shape[1], image.shape[0]))

                    # We crop only face detail for a better precision
                    # and accomodate landmark coordinates to cropped subimage
                    # as well as to float (0,1) scale
                    subimage, xmin, ymin, xmax, ymax = crop_face_only(image)
                    true_landmarks = np.subtract(true_landmarks, (xmin, image.shape[0] - ymax))
                    true_landmarks = np.divide(true_landmarks, (subimage.shape[1], subimage.shape[0]))
                    # As the source true landmarks have y-origin on the bottom of picture
                    # (unlike MediaPipe model)
                    # we have to flip their y-axis
                    true_landmarks[:,1] = 1 - true_landmarks[:,1]

                    if both_models:
                        input_landmarks = np.concatenate((MediaPipe_model(subimage), LBF_model(subimage)), axis = 0)
                    else:
                        input_landmarks = MediaPipe_model(subimage)
                    
                    angle = get_face_angle(input_landmarks, subimage.shape)
                    
                    input_landmarks = input_landmarks.reshape(1,-1)
                    x = np.concatenate((x, input_landmarks), axis = 0)

                    true_landmarks = true_landmarks.reshape(1,-1)
                    y_true = np.concatenate((y_true, true_landmarks), axis = 0)
                    
                    angles = np.concatenate((angles, np.array([angle])))
                    crops = np.concatenate((crops, np.array([xmin, ymin, xmax, ymax]).reshape(1,-1)), axis=0)
                    
                    path_list.append(img_path)
                    
                except Exception as e:
                    # Wrong image wasn't accepted into preprocessed data
                    print(f"Preprocessing went wrong in image: {img_path}, error: {e}.")

    return x, y_true, path_list, angles, crops



def save_preprocessed_data(x_inp, y_inp, path_list, angles, crops, target_folder):
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
        
    if os.path.exists(f'{target_folder}/path_list.txt'):
        already_preprocessed = np.load(f'{target_folder}/preprocessed_inputs.npz')
        already_saved_angles = np.load(f'{target_folder}/angles.npz')['angles']
        already_saved_crops = np.load(f'{target_folder}/crops.npz')['crops']
        all_inputs = already_preprocessed['x_inp']
        all_targets = already_preprocessed['y_inp']
        x_inp = np.concatenate((x_inp, all_inputs), axis = 0)
        y_inp = np.concatenate((y_inp, all_targets), axis = 0)
        angles = np.concatenate((angles, already_saved_angles))
        crops = np.concatenate((crops, already_saved_crops), axis = 0)
        
    np.savez(f'{target_folder}/preprocessed_inputs', x_inp = x_inp, y_inp = y_inp)
    np.savez(f'{target_folder}/angles', angles = angles)
    np.savez(f'{target_folder}/crops', crops = crops)
    
    path_file_exists = os.path.isfile(f'{target_folder}/path_list.txt')
    with open(f'{target_folder}/path_list.txt', 'a' if path_file_exists else 'w') as file:
        for path in path_list:
            file.write(path + '\n')


def preprocess_data(target_folder, data_folders):
    if not os.path.isfile(f"{target_folder}/path_list.txt"):  
        groups = get_sample_groups(data_folders)
        for group in tqdm(groups):
            x_inp, y_inp, path_list, angles, crops = prepare_training_landmarks(group=group)   
            # try:
            #     print(x_inp.shape)
            #     print(y_inp.shape)
            #     print(crops.shape)
            #     print(angles.shape)
            #     print(len(path_list)) 
            # except Exception as e:
            #     print(e)
                
            save_preprocessed_data(x_inp, y_inp, path_list, angles, crops, target_folder)
    else:
        print("Preprocessed data already exist - delete them first!")

# preprocess_data(target_folder="preprocessed_data", data_folders=['data'])