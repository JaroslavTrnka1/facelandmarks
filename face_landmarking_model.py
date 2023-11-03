from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from landmarks_utils import *
from cropping import *
from config import *



class RawProjection(nn.Module):
    def __init__(self, input_dim, output_dim, mask):
        super().__init__()

        #self.linear1 = nn.Linear(input_dim, input_dim, bias = False)
        self.linear2 = nn.Linear(input_dim, output_dim, bias = False)
        self.loss_func = nn.MSELoss()
        self.register_buffer('mask', mask)

    def forward(self, x, targets = None):

        #x = self.linear1(x)
        #self.linear1.weight.data = self.linear1.weight * self.mask

        output = F.linear(x, self.linear2.weight*self.mask, bias=None)

        if targets == None:
            loss = None
        else:
            loss = self.loss_func(output, targets)

        return output, loss

class EnsembleProjection(nn.Module):
    def __init__(self, input_dim, output_dim, num_projectors, projection_mask):
        super().__init__()
        
        self.num_projectors = num_projectors
        self.ensemble = nn.ModuleList()
        for i in range(num_projectors):
            projector = RawProjection(input_dim, output_dim, projection_mask)
            self.ensemble.append(projector)
    
    def forward(self, x, targets = None):
     
        outputs =  []
        losses = []
        
        if type(x) == list:
            pass
        else:
            x = [x] * self.num_projectors
            
        if type(targets) == list:
            pass
        else:
            targets = [targets] * self.num_projectors
        
        for i, projector in enumerate(self.ensemble):
            output, loss = projector(x[i], targets[i])
            outputs.append(output)
            
            losses.append(loss)
        
        return outputs, losses
    
    def predict(self, x):
        
        outputs = []
        
        for projector in self.ensemble:
            output, loss = projector(x)
            outputs.append(output)

        output = torch.mean(torch.stack(outputs, dim=0), dim=0)
        
        return output
    
    
# class MiniCNN(nn.Module):
#     def __init__(self, crop_size, hidden_per_lmark = 10):
#         super().__init__()

#         self.conv1 = nn.Conv2d(3, 3, 3, stride = 1, padding = 'same', padding_mode = 'replicate')
#         self.conv2 = nn.Conv2d(3, 6, 3, stride = 1, padding = 'same', padding_mode = 'replicate')
        
#         pool = 2
#         self.pool1 = nn.MaxPool2d((pool,pool))
#         self.pool2 = nn.MaxPool2d(3)

#         ch_per_lmark = self.conv2.weight.shape[0]
#         linear_input_dim = int((crop_size/pool)**2 * ch_per_lmark)
        
#         self.linear1 = nn.Linear(linear_input_dim, hidden_per_lmark)
#         self.linear2 = nn.Linear(hidden_per_lmark, 2)
        
#     def forward(self, x):

#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         #print(f"Shape after 1st pooling: {x.shape}")
#         x = F.relu(self.conv2(x))
#         #print(f"Shape after 2nd cnn: {x.shape}")
#         x = torch.flatten(x, 1)
#         #print(f"Shape after flattening: {x.shape}")

#         x = self.linear1(x)
#         output = self.linear2(x)

#         return output
    
# class CNNForlooping(nn.Module):
#     def __init__(self, crop_size, hidden_per_lmark=10):
#         super().__init__()
        
#         self.num_cnn = 72
#         self.multicnn = nn.ModuleList()
#         for i in range(self.num_cnn):
#             landmark_cnn = MiniCNN(crop_size, hidden_per_lmark)
#             self.multicnn.append(landmark_cnn)
        
#         self.crop_size = crop_size
#         self.grid = torch.stack(
#             torch.meshgrid(
#                 torch.arange(-crop_size//2, crop_size//2),
#                 torch.arange(-crop_size//2, crop_size//2)
#             )
#             , dim = 0).reshape(2,-1).unsqueeze(0).type(torch.int)

#     def forward(self, x, images):
        
#         # input x as a tensor of 144 landmark coordinates
#         # Pozor! images budou taky v batch!!2, takže shape: (batch, channels, height, width)
#         outputs = torch.empty(images.shape[0], 144)
#         images = images.clone().detach().permute(0,3,1,2)
        
#         for landmark, minicnn in enumerate(self.multicnn):
#             # shape (batch, 2)
#             central_pixels = x[:,landmark*2 : landmark*2+2] * torch.tensor([images.shape[-1], images.shape[-2]]).type(torch.int).to(DEVICE)
#             # shape (batch, 2, crop_size**2)
#             crop_field = central_pixels.unsqueeze(2) + self.grid
#             crop_field = crop_field.type(torch.int)
#             inbatch_idx = torch.arange(images.shape[0]).type(torch.int)
#             patches = images[:,:,crop_field[:,1,:], crop_field[:,0,:]][inbatch_idx,:,inbatch_idx,...].reshape(-1,3,self.crop_size,self.crop_size).type(torch.float)
#             output = minicnn(patches)
#             outputs[:,landmark*2 : landmark*2+2] = output

#         return outputs

class CNNFocusing(nn.Module):
    def __init__(self, crop_size, hidden_per_lmark=10):
        super().__init__()

        self.conv1 = nn.Conv2d(216, 216, 3, stride = 1, groups = 72, padding = 'same', padding_mode = 'replicate')
        self.conv2 = nn.Conv2d(216, 432, 3, stride = 1, groups = 72, padding = 'same', padding_mode = 'replicate')
        #self.conv3 = nn.Conv2d(72, 72, 3, stride = 1, groups = 72, padding = 'same', padding_mode = 'replicate')

        pool = 2
        self.pool1 = nn.MaxPool2d((pool,pool))
        #self.pool2 = nn.MaxPool2d(3)

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
    
    def forward(self, x):

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
    def __init__(self, projection_mask, projectors = 10):
        super().__init__()

        if BOTH_MODELS:
            input_dim = 1092
        else:
            input_dim = 956
            
        self.ensemble = EnsembleProjection(input_dim, 144, projectors, projection_mask[:, :input_dim])
        self.cnn_focusing = CNNFocusing(crop_size = CROP_SIZE, hidden_per_lmark=10)
        self.ffn = nn.Sequential(
          nn.Linear(288, 1024, bias=False),
          #nn.ReLU(),
          nn.Linear(1024, 144, bias=False),
          #nn.ReLU()
        )

        self.loss_function = nn.MSELoss()
        self.train_phase = 0


    def forward(self, x, targets, multicrop = None):

        if self.train_phase == 0:
            outputs, losses = self.ensemble(x, targets)
            # Loss_function už pouze na relative targets
            # abs_landmarks = get_absolute_positions(output.unflatten(-1, (-1, 2)), centroid, size_measure)
            # output = torch.flatten(abs_landmarks, start_dim = -2)
            final_loss = torch.sum(torch.stack(losses, dim=0), dim=0)

            return outputs, final_loss, None

        elif self.train_phase == 1:
            raw_landmarks = self.ensemble.predict(x)
            
            # abs_raw_landmarks = get_absolute_positions(raw_landmarks.unflatten(-1, (-1, 2)), centroid, size_measure)
            # abs_raw_landmarks = torch.flatten(abs_raw_landmarks, start_dim = -2)
            # raw_loss = self.pretraining_loss(abs_raw_landmarks, targets)
            raw_loss = self.loss_function(raw_landmarks, targets)

            correction = self.cnn_focusing(multicrop)

            output = raw_landmarks + correction
            # abs_landmarks = get_absolute_positions(output.unflatten(-1, (-1, 2)), centroid, size_measure)
            # output = torch.flatten(abs_landmarks, start_dim = -2)
            
            final_loss = self.loss_function(output, targets)

            return output, final_loss, raw_loss

        elif self.train_phase >= 2:
            raw_landmarks = self.ensemble.predict(x)
            raw_loss = self.loss_function(raw_landmarks, targets)
            
            correction = self.cnn_focusing(multicrop)

            ffn_input = torch.cat((raw_landmarks, correction), dim = 1)
            output = self.ffn(ffn_input) + raw_landmarks
            # abs_landmarks = get_absolute_positions(output.unflatten(-1, (-1, 2)), centroid, size_measure)
            # output = torch.flatten(abs_landmarks, start_dim = -2)
            
            final_loss = self.loss_function(output, targets)

            return output, final_loss, raw_loss    
    
    @torch.no_grad()
    def predict(self, image, face_detail = True):
        
        # TODO: Ošetřit multiple images
        # Preprocessing pathway
        if BOTH_MODELS:
            input_landmarks = np.concatenate((MediaPipe_model(image), LBF_model(image)), axis = 0)
        else:
            input_landmarks = MediaPipe_model(image)
        
        # Facedataset pathway    
        input_landmarks = torch.from_numpy(input_landmarks).float().to(DEVICE)
        angle = get_face_angle(input_landmarks, image.shape)
        relative_landmarks, centroid, size_measure = get_relative_positions(input_landmarks)
        subimage = crop_around_centroid(image, centroid, size_measure)
        subimage = standard_face_size(subimage)
        subimage = rotate_image(angle, subimage)  
        rotated_landmarks = rotate_landmarks(angle, relative_landmarks, subimage.shape)
        relative_landmarks = rotated_landmarks.reshape(1,-1)
        
        # Model pathway
        if self.train_phase == 0:
            output = self.ensemble.predict(relative_landmarks)
 
        elif self.train_phase == 1:
            raw_landmarks = self.ensemble.predict(relative_landmarks)
            multicrop = make_landmark_crops(raw_landmarks, subimage, crop_size = CROP_SIZE)
            correction = self.cnn_focusing(multicrop[None,:,:,:])

            output = raw_landmarks + correction
        
        elif self.train_phase == 2:
            raw_landmarks = self.ensemble.predict(relative_landmarks)
            multicrop = make_landmark_crops(raw_landmarks, subimage, crop_size = CROP_SIZE)
            correction = self.cnn_focusing(multicrop[None,:,:,:])
            ffn_input = torch.cat((raw_landmarks, correction), dim = 1)
            
            output = self.ffn(ffn_input) + raw_landmarks
        
        output = output.reshape(-1,2).detach()
        #output = output.unflatten(-1, (-1, 2)).cpu().detach()
        # output shape: (1, 72, 2) - důležité pro případný batching

        abs_output = get_absolute_positions(output, centroid, size_measure)    
        abs_output = abs_output.cpu().detach().numpy()
        output = output.numpy()
        
        # Pixel dimension
        # pixel_multiplyer = np.array([subimage.shape[1], subimage.shape[0]]).reshape(1,1,2)
        # output = np.multiply(output, pixel_multiplyer).astype(np.int32)
        # abs_output = np.multiply(abs_output, pixel_multiplyer).astype(np.int32)
        
        if face_detail:
            return output, relative_landmarks, subimage
        
        else:
            return abs_output, input_landmarks, image