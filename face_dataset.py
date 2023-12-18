import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import cv2
import numpy as np

from cropping import standard_face_size, make_landmark_crops, crop_around_centroid, rotate_image, get_subimage_shape, normalize_multicrop, template_matching
from config import *
from landmarks_utils import *


class FaceDataset(Dataset):
    def __init__(self, model=None,
                 subgroups = None,
                 rotate = True,
                 avg_template = None,
                 template_method = None,
                 work = False,
                 gray = False,
                 crop_as_template = False
                 ):
   
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
            self.x = subgroup_x[:, :956]
            
        self.y_true = subgroup_y_true
        self.angles = subgroup_angles
        self.path_list = subgroup_pathlist
          
        self.model = model
        self.rotate = rotate
        self.template = avg_template
        self.template_method = eval(str(template_method))
        self.work = work
        self.gray = gray
        self.crop_as_template = crop_as_template
        
        if self.template is not None:
            self.pretraining = True
        else:
            self.pretraining = False
            

    def __len__(self):
        return len(self.path_list)

    
    def __getitem__(self, idx):
        
        x = torch.tensor(self.x[idx,:], dtype = torch.float).to(DEVICE)
        y = torch.tensor(self.y_true[idx,:], dtype = torch.float).to(DEVICE)
        img_path = self.path_list[idx]
        
        relative_landmarks, centroid, size_measure = get_relative_positions(x.reshape(-1,2))
        relative_targets = fit_to_relative_centroid(y.reshape(-1,2), centroid, size_measure)
        subimage_shape = get_subimage_shape(img_path, size_measure)
        
        if self.rotate:
            angle = self.angles[idx]
            relative_landmarks = rotate_landmarks(angle, relative_landmarks, subimage_shape)
            relative_targets = rotate_landmarks(angle, relative_targets, subimage_shape)
            
        x = relative_landmarks.reshape(x.shape)
        y = relative_targets.reshape(y.shape)

        if self.pretraining and self.template is not None:
            return x, y, 0

        else:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            subimage = crop_around_centroid(image, centroid, size_measure)
            subimage = standard_face_size(subimage)
            
            if self.rotate:
                subimage = rotate_image(angle, subimage)
                
            # img_edges = get_image_edges(subimage)[:,:,None]
            
            # # In case of template preparation_dataset
            # if self.template is None:
            #     return y, img_edges, img_path
            
            raw_landmarks = self.model.ensemble.predict(x)
            
            if self.gray:
                subimage = cv2.cvtColor(subimage, cv2.COLOR_BGR2GRAY)[:,:,None]
                
            multicrop = normalize_multicrop(make_landmark_crops(raw_landmarks, subimage, CROP_SIZE))
            # multicrop = make_landmark_crops(raw_landmarks, img_edges, CROP_SIZE)
            # multicrop = make_landmark_crops(y, subimage, CROP_SIZE)

            template_match = template_matching(multicrop, self.template, self.template_method, crop_as_template=self.crop_as_template)
            
        if not self.work:
            return x, y, template_match #, multicrop, subimage, image
        else:
            return x, y, multicrop, subimage , image#, template_match
    
    def get_landmarks(self, idx):
        x = torch.tensor(self.x[idx,:], dtype = torch.float).to(DEVICE)
        y = torch.tensor(self.y_true[idx,:], dtype = torch.float).to(DEVICE)
        img_path = self.path_list[idx]
        
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
        
    def get_gray_multicrop(self, idx, crop_size, gray = True, from_targets = True, work_with_image = False):
        
        x = torch.tensor(self.x[idx,:], dtype = torch.float).to(DEVICE)
        y = torch.tensor(self.y_true[idx,:], dtype = torch.float).to(DEVICE)
        img_path = self.path_list[idx]
        
        if from_targets:
            relative_targets, centroid, size_measure = get_relative_positions(y.reshape(-1,2))
        else:
            relative_landmarks, centroid, size_measure = get_relative_positions(x.reshape(-1,2))
            relative_targets = fit_to_relative_centroid(y.reshape(-1,2), centroid, size_measure)
            
        subimage_shape = get_subimage_shape(img_path, size_measure)
        
        if self.rotate: 
            angle = self.angles[idx]
            relative_targets = rotate_landmarks(angle, relative_targets, subimage_shape)
            if not from_targets:
                relative_landmarks = rotate_landmarks(angle, relative_landmarks, subimage_shape)
                
        if not from_targets:
            x = relative_landmarks.reshape(x.shape)
        y = relative_targets.reshape(y.shape)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        subimage = crop_around_centroid(image, centroid, size_measure)
        subimage = standard_face_size(subimage)
        
        if self.rotate:
            subimage = rotate_image(angle, subimage)
            
        # img_edges = get_image_edges(subimage)[:,:,None]
        
        if gray:       
            subimage = cv2.cvtColor(subimage, cv2.COLOR_BGR2GRAY)[:,:,None]

        if from_targets:
            multicrop = make_landmark_crops(y, subimage, crop_size)
        else:
            multicrop = make_landmark_crops(x, subimage, crop_size)
            
        if not work_with_image:
            return x, y, multicrop
        else:
            return x, y, multicrop, subimage , image   
        
    # def __getitem__(self, idx):
        
    #     x = torch.tensor(self.x[idx,:], dtype = torch.float).to(DEVICE)
    #     y = torch.tensor(self.y_true[idx,:], dtype = torch.float).to(DEVICE)
    #     img_path = self.path_list[idx]
        
    #     relative_landmarks, centroid, size_measure = get_relative_positions(x.reshape(-1,2))
    #     relative_targets = fit_to_relative_centroid(y.reshape(-1,2), centroid, size_measure)
    #     subimage_shape = get_subimage_shape(img_path, size_measure)
        
    #     if self.rotate:
    #         angle = self.angles[idx]
    #         relative_landmarks = rotate_landmarks(angle, relative_landmarks, subimage_shape)
    #         relative_targets = rotate_landmarks(angle, relative_targets, subimage_shape)
            
    #     x = relative_landmarks.reshape(x.shape)
    #     y = relative_targets.reshape(y.shape)

    #     if self.pretraining and self.template is not None:
    #         return x, y, 0

    #     else:
    #         image = cv2.imread(img_path)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #         subimage = crop_around_centroid(image, centroid, size_measure)
    #         subimage = standard_face_size(subimage)
            
    #         if self.rotate:
    #             subimage = rotate_image(angle, subimage)
                
    #         img_edges = get_image_edges(subimage)[:,:,None]
            
    #         # In case of template preparation_dataset
    #         if self.template is None:
    #             return y, img_edges, img_path
            
    #         raw_landmarks = self.model.ensemble.predict(x)
            
    #         # multicrop = make_landmark_crops(raw_landmarks, subimage, CROP_SIZE)
    #         #multicrop = make_landmark_crops(raw_landmarks, img_edges, CROP_SIZE)
    #         multicrop = make_landmark_crops(y, subimage, CROP_SIZE)
            
    #         template_match = template_matching(multicrop, self.template, self.template_method)
            
    #     if not self.work:
    #         return x, y, multicrop#, multicrop, subimage, image
    #     else:
    #         return x, y, multicrop, subimage , image#, template_match
        
    
class EnsembleSampler:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(torch.randperm(len(self.dataset))[:200])

    def __len__(self):
        return len(self.dataset)