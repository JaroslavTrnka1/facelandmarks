import torch
#import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np

from cropping import standard_face_size, make_landmark_crops
from config import *
from landmarks_utils import *


class FaceDataset(Dataset):
    def __init__(self, model):
   
        preprocessed_inputs = np.load('preprocessed_data/preprocessed_inputs.npz')
        path_list = []
                
        with open("preprocessed_data/path_list.txt", "r") as pl:
            for path in pl:
                path_list.append(path.strip())

        self.x = preprocessed_inputs['x_inp']
        self.y_true = preprocessed_inputs['y_inp']
        self.path_list = path_list
        self.pretraining = True
        self.model = model

    def __len__(self):
        return len(self.path_list)

    
    def __getitem__(self, idx):
        
        x = torch.tensor(self.x[idx,:], dtype = torch.float).to(DEVICE)
        y = torch.tensor(self.y_true[idx,:], dtype = torch.float).to(DEVICE)
        
        relative_landmarks, centroid, size_measure = get_relative_positions(x.reshape(-1,2))
        x = relative_landmarks.reshape(x.shape)

        if self.pretraining:
            multicrop = 0

        else:
            img_path = self.path_list[idx]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            subimage_center = torch.mul(centroid, torch.tensor([image.shape[1], image.shape[0]])).type(torch.int)
            subimage_size = torch.mul(size_measure, torch.tensor([image.shape[1], image.shape[0]])).type(torch.int)
            subimage_from = torch.squeeze(subimage_center - subimage_size)
            subimage_to = torch.squeeze(subimage_center + subimage_size)
            
            subimage = np.ascontiguousarray(image[subimage_from[1]:subimage_to[1], subimage_from[0]:subimage_to[0], :])
            subimage = standard_face_size(subimage)
            
            raw_landmarks, _ = self.model.raw_projection(x)
            multicrop = make_landmark_crops(raw_landmarks, subimage, CROP_SIZE)

        return x, y, centroid, size_measure, multicrop