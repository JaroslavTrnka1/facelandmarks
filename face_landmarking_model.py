import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from landmarks_utils import MediaPipe_model, LBF_model
from cropping import make_landmark_crops, crop_face_only, standard_face_size

from config import *



class RawProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, input_dim, bias = True)
        self.linear2 = nn.Linear(input_dim, output_dim, bias = True)
        self.loss_func = nn.MSELoss()

    def forward(self, x, targets = None):

        x = self.linear1(x)
        output = self.linear2(x)

        if targets == None:
            loss = None
        else:
            loss = self.loss_func(output, targets)

        return output, loss
    
class CNNFocusing(nn.Module):
    def __init__(self, crop_size):
        super().__init__()

        self.conv1 = nn.Conv2d(216, 216, 3, stride = 1, groups = 72, padding = 'same', padding_mode = 'replicate')
        self.conv2 = nn.Conv2d(216, 432, 3, stride = 1, groups = 72, padding = 'same', padding_mode = 'replicate')
        #self.conv3 = nn.Conv2d(72, 72, 3, stride = 1, groups = 72, padding = 'same', padding_mode = 'replicate')

        pool = 2
        self.pool1 = nn.MaxPool2d((pool,pool))
        self.pool2 = nn.MaxPool2d(3)

        hidden_per_lmark = 10
        ch_per_lmark = self.conv2.weight.shape[0] // 72
        hidden_dim = int((crop_size/pool)**2 * ch_per_lmark)
        mask_diag = torch.diag(torch.ones(72))
        linear1_mask = mask_diag.repeat_interleave(hidden_per_lmark, dim = 1).repeat_interleave(hidden_dim, dim = 0)
        linear2_mask = mask_diag.repeat_interleave(2, dim = 1).repeat_interleave(hidden_per_lmark, dim = 0)

        self.register_buffer("mask1", linear1_mask)
        self.register_buffer("mask2", linear2_mask)

        self.linear1 = nn.Linear(linear1_mask.shape[0], linear1_mask.shape[1])
        self.linear2 = nn.Linear(linear2_mask.shape[0], linear2_mask.shape[1])

        self.crop_size = crop_size

    def forward(self, x, image_shape = None):

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)


        self.linear1.weight.data.mul_(self.mask1.permute(1,0))
        x = self.linear1(x)

        self.linear2.weight.data.mul_(self.mask2.permute(1,0))
        output = self.linear2(x)

        return output
    
class FaceLandmarking(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_projection = RawProjection(956, 144)
        self.cnn_focusing = CNNFocusing(crop_size = CROP_SIZE)

        self.ffn = nn.Sequential(
          nn.Linear(288, 1024),
          #nn.ReLU(),
          nn.Linear(1024, 144),
          #nn.ReLU()
        )

        self.final_loss = nn.L1Loss()
        self.pretraining_loss = nn.MSELoss()
        self.train_phase = 0

    def forward(self, x, targets, multicrop = None, image_shape = None):

        if self.train_phase == 0:

            output, _ = self.raw_projection(x, targets)
            final_loss = self.pretraining_loss(output, targets)

            return output, final_loss, None

        elif self.train_phase == 1:
            raw_landmarks, raw_loss = self.raw_projection(x, targets)
            correction = self.cnn_focusing(multicrop, image_shape)

            output = raw_landmarks + correction
            final_loss = self.pretraining_loss(output, targets)

            return output, final_loss, raw_loss

        elif self.train_phase == 2:
            raw_landmarks, raw_loss = self.raw_projection(x, targets)
            correction = self.cnn_focusing(multicrop, image_shape)

            ffn_input = torch.cat((raw_landmarks, correction), dim = 1)
            output = self.ffn(ffn_input) + raw_landmarks
            final_loss = self.pretraining_loss(output, targets)

            return output, final_loss, raw_loss
    
    # bude třeba předělat na crop face only
    @torch.no_grad()
    def predict(self, image, face_detail = True):
        try:
            subimage, xmin, ymin, xmax, ymax = crop_face_only(image)
        except:
            print('No face has been found in the image!')
            return None
        
        if BOTH_MODELS:
            input_landmarks = np.concatenate((LBF_model(subimage), MediaPipe_model(subimage)), axis = 0)
        else:
            input_landmarks = MediaPipe_model(subimage)
            
        input_landmarks = torch.from_numpy(input_landmarks.reshape(1,-1)).float().to(DEVICE)
        
        raw_landmarks, _ = self.raw_projection(input_landmarks)
        
        subimage = standard_face_size(subimage)
        multicrop = make_landmark_crops(raw_landmarks, subimage, crop_size = CROP_SIZE)
        
        correction = self.cnn_focusing(multicrop[None,:,:,:], image_shape = subimage.shape)
        ffn_input = torch.cat((raw_landmarks, correction), dim = 1)
        
        output = self.ffn(ffn_input) + raw_landmarks
        output = output.cpu().detach().numpy().reshape(-1,2)
        
        # Flipping the axis to have the origin bottom-left
        output[:,1] = 1 - output[:,1]
        
        # Pixel dimension
        output = np.multiply(output, (subimage.shape[1], subimage.shape[0]))
        
        if face_detail:
            return output, subimage
        
        else:
            output = np.add(output, (xmin, image.shape[0] - ymax))
            return output, image