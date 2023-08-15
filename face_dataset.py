import torch
#import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np

from cropping import standard_face_size, make_landmark_crops
from config import *


class FaceDataset(Dataset):
    def __init__(self, model):
   
        preprocessed_inputs = np.load('preprocessed_data/preprocessed_inputs.npz')
        face_detail_coordinates = np.load('preprocessed_data/face_detail_coordinates.npz')
        path_list = []
                
        with open("preprocessed_data/path_list.txt", "r") as pl:
            for path in pl:
                path_list.append(path.strip())

        self.x = preprocessed_inputs['x_inp']
        self.y_true = preprocessed_inputs['y_inp']
        self.path_list = path_list
        self.face_detail_coordinates = face_detail_coordinates['fdc'].astype(int)
        self.pretraining = True
        self.model = model

    def __len__(self):
        return len(self.path_list)

    
    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx,:], dtype = torch.float).to(DEVICE)
        y = torch.tensor(self.y_true[idx,:], dtype = torch.float).to(DEVICE)

        if self.pretraining:
            multicrop = 0

        else:
            img_path = self.path_list[idx]
            xmin, ymin, xmax, ymax = self.face_detail_coordinates[idx,:]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.ascontiguousarray(image[ymin:ymax, xmin:xmax, :])
            image = standard_face_size(image)
            
            raw_landmarks, _ = self.model.raw_projection(x, None)
            multicrop = make_landmark_crops(raw_landmarks, image)

        return x, y, multicrop