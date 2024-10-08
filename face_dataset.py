import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import cv2
import numpy as np

from facelandmarks.cropping import standard_face_size, make_landmark_crops, crop_around_centroid, rotate_image, get_subimage_shape, normalize_multicrop, create_true_heatmaps
from facelandmarks.config import *
from facelandmarks.landmarks_utils import *


class FaceDataset(Dataset):
    def __init__(self, model=None,
                 subgroups = None,
                 rotate = True,
                 work = False,
                 gray = False,
                 crop_size = 46
                 ):
   
        preprocessed_inputs = np.load('preprocessed_data/preprocessed_inputs.npz')
        all_angles = np.load('preprocessed_data/angles.npz')['angles']
        all_crops = np.load('preprocessed_data/crops.npz')['crops']
        path_list = []
                
        with open("preprocessed_data/path_list.txt", "r") as pl:
            for path in pl:
                path_list.append(path.strip().replace('\\', '/'))
        
        all_inputs = preprocessed_inputs['x_inp']
        all_targets = preprocessed_inputs['y_inp']
        # print(f'Inputs and targets has the same length: {all_inputs.shape[0] == all_targets.shape[0]}.')
        # print(f'Crops and path_list have equal length: {len(path_list) == all_crops.shape[0]}')
        
        if subgroups:
            subgroups = list(subgroups)
            subgroup_x = np.empty((0, all_inputs.shape[1]))
            subgroup_y_true = np.empty((0, all_targets.shape[1]))
            subgroup_angles = np.empty(0)
            subgroup_crops = np.empty((0,4))
            subgroup_pathlist = []
            
            for subgroup in subgroups:
                for i, path in enumerate(path_list):
                    if subgroup in path:
                        subgroup_pathlist.append(path)
                        subgroup_x = np.concatenate((subgroup_x, all_inputs[i:i+1,:]), axis = 0)
                        subgroup_y_true = np.concatenate((subgroup_y_true, all_targets[i:i+1,:]), axis = 0)
                        subgroup_angles = np.concatenate([subgroup_angles, all_angles[i:i+1]], axis = 0)
                        subgroup_crops = np.concatenate([subgroup_crops, all_crops[i:i+1,:]], axis = 0)
        else:
            subgroup_x = all_inputs
            subgroup_y_true = all_targets
            subgroup_pathlist = path_list
            subgroup_angles = all_angles
            subgroup_crops = all_crops
        
        if BOTH_MODELS:
            self.x = subgroup_x
        else:
            self.x = subgroup_x[:, :956]
            
        self.y_true = subgroup_y_true
        self.angles = subgroup_angles
        self.crops = subgroup_crops
        self.path_list = subgroup_pathlist
        self.crop_size = crop_size  
        self.model = model
        self.rotate = rotate
        self.pretraining = True
        self.work = work
        self.gray = gray
   
             
    def __len__(self):
        return len(self.path_list)

    
    def __getitem__(self, idx):
        
        x = torch.tensor(self.x[idx,:], dtype = torch.float)
        y = torch.tensor(self.y_true[idx,:], dtype = torch.float)
        img_path = self.path_list[idx]
        # crops = torch.tensor(self.crops[idx], dtype = torch.int32)
        
        relative_landmarks, centroid, size_measure = get_relative_positions(x.reshape(-1,2))
        relative_targets = fit_to_relative_centroid(y.reshape(-1,2), centroid, size_measure)
        subimage_shape = get_subimage_shape(img_path, size_measure)
        # subimage_shape = torch.tensor([crops[3] - crops[1], crops[2] - crops[0]])
        
        if self.rotate:
            angle = self.angles[idx]
            relative_landmarks = rotate_landmarks(angle, relative_landmarks, subimage_shape)
            relative_targets = rotate_landmarks(angle, relative_targets, subimage_shape)
            
        x = relative_landmarks.reshape(x.shape).type(torch.float32)
        y = relative_targets.reshape(y.shape).type(torch.float32)
        
        if self.pretraining:
            return x, y, 0

        else:
            if os.path.exists(img_path):
                image = cv2.imread(img_path)
            else:
                image = cv2.imread('.'.join(img_path.split('.')[:-1]) + '.' + img_path.split('.')[-1].upper())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # subimage_first = np.ascontiguousarray(image[crops[1]:crops[3], crops[0]:crops[2], :])

            subimage = crop_around_centroid(image, centroid, size_measure)
            subimage = standard_face_size(subimage)
            
            if self.rotate:
                subimage = rotate_image(angle, subimage)
            
            raw_landmarks = self.model.ensemble.predict(x)
            
            if self.gray:
                subimage = cv2.cvtColor(subimage, cv2.COLOR_BGR2GRAY)[:,:,None]
            
            multicrop = normalize_multicrop(make_landmark_crops(raw_landmarks.to('cpu'), subimage, self.crop_size))
            
            # Ve chvíli, kdy půjde o heatmaps, bude třeba vracet target heatmaps, image_size
            # tedy:x, heatmaps, multicrop, raw_landmarks 

            # multicrop = make_landmark_crops(y, subimage, self.crop_size)
            if self.model.train_phase == 1:
                if self.work:
                    return x, y, multicrop, image, subimage
                
                else:
                    if self.model.heatmaps:
                        heatmaps = create_true_heatmaps(raw_landmarks.to('cpu'), y, subimage, self.crop_size)
                    else:
                        heatmaps = 0
                    
                    # print(x.shape, y.shape, multicrop.shape, raw_landmarks.shape)    
                    return x, y, multicrop, raw_landmarks, heatmaps, torch.tensor([subimage.shape[1], subimage.shape[0]])
            
            elif self.model.train_phase == 2:
                catenate = (self.model.top_head == 'ffn-catenate')
                correction = self.model.cnn_ensemble.predict(multicrop, catenate=catenate)
                if catenate:
                    landmarks = torch.cat((raw_landmarks, correction), dim = 1)
                    multicrop2 = multicrop
                else:
                    landmarks  = raw_landmarks + correction
                    multicrop2 = normalize_multicrop(make_landmark_crops(landmarks, subimage, self.crop_size))
                return x, y, multicrop2, landmarks
                
    def get_landmarks(self, idx):
        x = torch.tensor(self.x[idx,:], dtype = torch.float)
        y = torch.tensor(self.y_true[idx,:], dtype = torch.float)
        img_path = self.path_list[idx]
        if not os.path.exists(img_path):
            img_path = '.'.join(img_path.split('.')[:-1]) + '.' + img_path.split('.')[-1].upper()
        
        relative_landmarks, centroid, size_measure = get_relative_positions(x.reshape(-1,2))
        relative_targets = fit_to_relative_centroid(y.reshape(-1,2), centroid, size_measure)
            
        subimage_shape = get_subimage_shape(img_path, size_measure)
        
        if self.rotate: 
            angle = self.angles[idx]
            relative_landmarks = rotate_landmarks(angle, relative_landmarks, subimage_shape)
            relative_targets = rotate_landmarks(angle, relative_targets, subimage_shape)
            
        x = relative_landmarks.reshape(x.shape)
        y = relative_targets.reshape(y.shape)  
        
        return x, y     
        
    
class EnsembleSampler:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(torch.randperm(len(self.dataset))[:200])

    def __len__(self):
        return len(self.dataset)