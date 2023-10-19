import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import cv2
import numpy as np

from cropping import standard_face_size, make_landmark_crops, crop_around_centroid, rotate_image, get_subimage_shape
from config import *
from landmarks_utils import *


class FaceDataset(Dataset):
    def __init__(self, model, subgroups = None, rotate = True):
   
        preprocessed_inputs = np.load('preprocessed_data/preprocessed_inputs.npz')
        all_angles = np.load('preprocessed_data/angles.npz')
        path_list = []
                
        with open("preprocessed_data/path_list.txt", "r") as pl:
            for path in pl:
                path_list.append(path.strip())
        
        all_inputs = preprocessed_inputs['x_inp']
        all_targets = preprocessed_inputs['y_inp']

        
        if subgroups:
            subgroups = list(subgroups)
            subgroup_x = np.empty((0, all_inputs.shape[1]))
            subgroup_y_true = np.empty((0, all_targets.shape[1]))
            subgroup_angles = np.empty(0)
            subgroup_pathlist = []
            
            for subgroup in subgroups:
                for i, path in enumerate(path_list):
                    if subgroup in path:
                        subgroup_pathlist.append(path)
                        subgroup_x = np.concatenate((subgroup_x, all_inputs[i:i+1,:]), axis = 0)
                        subgroup_y_true = np.concatenate((subgroup_y_true, all_targets[i:i+1,:]), axis = 0)
                        subgroup_angles = np.concatenate([subgroup_angles, all_angles['angles'][i:i+1]], axis = 0)
        else:
            subgroup_x = all_inputs
            subgroup_y_true = all_targets
            subgroup_pathlist = path_list
            subgroup_angles = all_angles['angles']
        
        if BOTH_MODELS:
            self.x = subgroup_x
        else:
            self.x = subgroup_x[:, :model.ensemble.ensemble[0].linear2.weight.shape[1]]
            
        self.y_true = subgroup_y_true
        self.angles = subgroup_angles
        self.path_list = subgroup_pathlist
        self.pretraining = True
        self.model = model
        self.rotate = rotate

    def __len__(self):
        return len(self.path_list)

    
    def __getitem__(self, idx):
        
        x = torch.tensor(self.x[idx,:], dtype = torch.float).to(DEVICE)
        y = torch.tensor(self.y_true[idx,:], dtype = torch.float).to(DEVICE)
        img_path = self.path_list[idx]
        
        if self.rotate:
            angle = self.angles[idx]
        else:
            angle = 0
               
        relative_landmarks, centroid, size_measure = get_relative_positions(x.reshape(-1,2))
        subimage_shape = get_subimage_shape(img_path, size_measure)
        rotated_landmarks = rotate_landmarks(angle, relative_landmarks, subimage_shape)
        x = rotated_landmarks.reshape(x.shape)
        
        relative_targets = fit_to_relative_centroid(y.reshape(-1,2), centroid, size_measure)
        rotated_targets = rotate_landmarks(angle, relative_targets, subimage_shape)
        y = rotated_targets.reshape(y.shape)
        #subimage = 0

        if self.pretraining:
            multicrop = 0

        else:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            subimage = crop_around_centroid(image, centroid, size_measure)
            subimage = standard_face_size(subimage)
            subimage = rotate_image(angle, subimage)
            
            raw_landmarks = self.model.ensemble.predict(x)
            multicrop = make_landmark_crops(raw_landmarks, subimage, CROP_SIZE)

        return x, y, multicrop #, subimage

class EnsembleSampler:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(torch.randperm(len(self.dataset))[:200])

    def __len__(self):
        return len(self.dataset)