from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from facelandmarks.landmarks_utils import *
from facelandmarks.cropping import *
from facelandmarks.config import *
from facelandmarks.transformer_focusing import GroupedTransformer




class WingLoss(nn.Module):
    def __init__(self, width=5, curvature=0.5):
        super(WingLoss, self).__init__()
        self.width = width
        self.curvature = curvature
        self.C = self.width - self.width * np.log(1 + self.width / self.curvature)

    def forward(self, prediction, target):
        diff = target - prediction
        diff_abs = diff.abs()
        loss = diff_abs.clone()

        idx_smaller = diff_abs < self.width
        idx_bigger = diff_abs >= self.width

        loss[idx_smaller] = self.width * torch.log(1 + diff_abs[idx_smaller] / self.curvature)
        loss[idx_bigger]  = loss[idx_bigger] - self.C
        loss = loss.mean()
        return loss

class RawProjection(nn.Module):
    def __init__(self, input_dim, output_dim, mask):
        super().__init__()

        #self.linear1 = nn.Linear(input_dim, input_dim, bias = False)
        self.linear2 = nn.Linear(input_dim, output_dim, bias = False)
        self.loss_func = nn.MSELoss()
        
        if mask is not None:
            self.register_buffer('mask', mask)
        else:
            self.register_buffer('mask', torch.empty([144, 956]))

    def forward(self, x, targets = None):

        #x = self.linear1(x)
        #self.linear1.weight.data = self.linear1.weight * self.mask
        if str(x.device).split(':')[0] != str(DEVICE):
            # print(f'moving from{x.device} to {DEVICE}')
            x = x.to(DEVICE)
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
        if projection_mask is not None:
            projection_mask = projection_mask[:, :input_dim]
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
    
    @torch.no_grad()
    def predict(self, x):
        
        outputs = []
        
        for projector in self.ensemble:
            output, loss = projector(x)
            outputs.append(output)

        output = torch.mean(torch.stack(outputs, dim=0), dim=0)
        
        return output

class CNNFocusing(nn.Module):
    def __init__(self, kernel_sizes, activations = True, pooling = True, batch_norm = False, crop_size = 46, start_out_channels = 8, hidden_per_lmark = 64):
        super().__init__()
        
        self.cnn = self.initialize_cnn(kernel_sizes, activations, pooling, start_out_channels=start_out_channels, batch_norm=batch_norm)
        cnn_output_size = self.calculate_output_size(crop_size, self.cnn)
        self.crop_size = crop_size
        mask_diag = torch.diag(torch.ones(72))
        linear1_mask = mask_diag.repeat_interleave(hidden_per_lmark, dim = 1).repeat_interleave(int(cnn_output_size), dim = 0)
        linear2_mask = mask_diag.repeat_interleave(2, dim = 1).repeat_interleave(hidden_per_lmark, dim = 0)

        self.register_buffer("mask1", linear1_mask)
        self.register_buffer("mask2", linear2_mask)
        
        self.linear1 = nn.Linear(linear1_mask.shape[0], linear1_mask.shape[1], bias = False)
        self.linear2 = nn.Linear(linear2_mask.shape[0], linear2_mask.shape[1])
    
    def forward(self, x):
        if x.shape[-1] != self.crop_size:
            new_crop = x.shape[-1]
            
            x = x[...,int((new_crop - self.crop_size)/2): int((new_crop + self.crop_size)/2), int((new_crop - self.crop_size)/2): int((new_crop + self.crop_size)/2)]
        x = self.cnn(x)
        x = torch.flatten(x, -3)
        self.linear1.weight.data.mul_(self.mask1.permute(1,0))
        output = self.linear1(x)
        self.linear2.weight.data.mul_(self.mask2.permute(1,0))
        output = self.linear2(output)
        return output
        
    def initialize_cnn(self, kernel_sizes: list, activations = False, pooling = True, batch_norm = False, start_out_channels = 8):
        if not isinstance(pooling, list):
            pooling = [pooling] * len(kernel_sizes)
        if not isinstance(activations, list):
            activations = [activations] * len(kernel_sizes)
        cnn_layers = []
        in_channels = 3 
        out_channels = start_out_channels
        for kernel_size, activation, pool in zip(kernel_sizes, activations, pooling):
            cnn_layers.append(nn.Conv2d(in_channels * 72, out_channels * 72, kernel_size, padding=0, groups=72))
            if batch_norm:
                # cnn_layers.append(nn.GroupNorm(72, out_channels * 72))
                cnn_layers.append(nn.BatchNorm2d(out_channels * 72))
            if activation:
                cnn_layers.append(nn.ReLU())
            if pool:
                if pool == 'avg':
                    cnn_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
                else:
                    cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_channels = out_channels
            out_channels = 2* out_channels

        cnn_model = nn.Sequential(*cnn_layers)
        return cnn_model   
    
    def calculate_layer_output_size(self, input_size, kernel_size, padding, stride):
        return ((input_size - kernel_size + 2 * padding) // stride) + 1

    def calculate_output_size(self, input_size, model):
        # Iterate through the convolutional layers in the model
        for layer in model.children():
            if isinstance(layer, nn.Conv2d):
                input_size = self.calculate_layer_output_size(input_size, layer.kernel_size[0], layer.padding[0], layer.stride[0])
                out_channels = layer.out_channels
            elif isinstance(layer, nn.AvgPool2d) or isinstance(layer, nn.MaxPool2d):
                # Adjust input size for max pooling layer
                input_size = self.calculate_layer_output_size(input_size, layer.kernel_size, layer.padding, layer.stride)
        
        # print(f'({out_channels} * {input_size} * {input_size} / 72)')        
        input_size = out_channels * input_size**2 / 72
        return input_size 
            

class EnsembleCNN(nn.Module):
    def __init__(self, num_cnn, kernel_sizes, activations, pooling, batch_norm, crop_size, start_out_channels, hidden_per_lmark, mixture = False):
        super().__init__()
        
        self.ensembleCNN = nn.ModuleList()
        self.num_cnn = num_cnn
        
        if not isinstance(crop_size, list):
            crop_size = [crop_size] * num_cnn
        
        for i in range(num_cnn):
            cnn_focusing = CNNFocusing(
                kernel_sizes=kernel_sizes,
                activations=activations,
                pooling=pooling[i],
                batch_norm=batch_norm,
                crop_size = crop_size[i],
                start_out_channels=start_out_channels,
                hidden_per_lmark = hidden_per_lmark
                )
            self.ensembleCNN.append(cnn_focusing)
        
        if mixture:
            self.router = nn.Parameter(torch.ones((144, self.num_cnn)))
        else:
            self.router = None
        
    def forward(self, x, catenate = True):
        if self.router is not None:
            normalized_router = torch.nn.functional.softmax(self.router, dim=-1)
            weights = normalized_router[:,0]
            output = torch.mul(self.ensembleCNN[0](x), weights)
            for i, cnn in enumerate(self.ensembleCNN[1:]):
                weights = normalized_router[:,i+1]
                cnn_out = torch.mul(cnn(x), weights)
                if catenate:
                    output = torch.cat([output, cnn_out], dim = 1)
                else:
                    output += cnn_out
        else:
            output = self.ensembleCNN[0](x)
            for cnn in self.ensembleCNN[1:]:
                if catenate:
                    output = torch.cat([output, cnn(x)], dim=1)
                else:
                    output += cnn(x)
        return output
    
    @torch.no_grad()
    def predict(self, x, catenate = True):
        output = self.ensembleCNN[0](x)
        for cnn in self.ensembleCNN[1:]:
            if catenate:
                output = torch.cat([output, cnn(x)], dim = 1)
            else:
                output += cnn(x)
        return output
   
class FaceLandmarking(nn.Module):
    def __init__(self, 
                 projection_mask = None, 
                 projectors = 3,
                 kernel_sizes = [3,3,3],
                 activations = False,
                 pooling = ['max', 'avg'],
                 crop_size = 46,
                 start_out_channels=8,
                 ffn_projection_mask = None,
                 batch_norm = False,
                 num_cnn = 4,
                 wing_loss_width=5,
                 hidden_per_lmark=64,
                 mixture = None
                 ):

        self.args = locals()
        del self.args['self'] 
        del self.args['__class__']
        
        super().__init__()

        if BOTH_MODELS:
            input_dim = 1092
        else:
            input_dim = 956
            
        self.ensemble = EnsembleProjection(input_dim, 144, projectors, projection_mask)

        cnn_pooling = []
        while len(cnn_pooling) < num_cnn:
            cnn_pooling.extend(pooling)
            cnn_pooling = cnn_pooling[:num_cnn]
        if FOCUSING == 'CNN':
            self.cnn_ensemble = EnsembleCNN(num_cnn=num_cnn,
                                            kernel_sizes=kernel_sizes,
                                            activations=activations,
                                            pooling=cnn_pooling,
                                            batch_norm=batch_norm,
                                            crop_size = crop_size,
                                            start_out_channels=start_out_channels,
                                            hidden_per_lmark=hidden_per_lmark,
                                            mixture=mixture)
        else:
            self.transformer_focusing = GroupedTransformer(crop_size=crop_size,
                                                        num_crops=72,
                                                        patch_size=5,
                                                        embed_dim=64,
                                                        num_heads=4,
                                                        num_blocks=4)
            # self.cnn_ensemble2 = EnsembleCNN(num_cnn=num_cnn,
            #                                 kernel_sizes=kernel_sizes,
            #                                 activations=activations,
            #                                 pooling=cnn_pooling,
            #                                 batch_norm=batch_norm,
            #                                 crop_size = crop_size,
            #                                 start_out_channels=start_out_channels)
            
        # self.ffn = nn.Sequential(
        #   nn.Linear(144, 288 * 2, bias=False),
        #   nn.Tanh(),
        #   nn.Linear(288 * 2, 144, bias=False),
        # )
        self.crop_size = crop_size
        if ffn_projection_mask is not None:
            self.register_buffer('ffn_projection_mask', ffn_projection_mask)
        else:
            self.register_buffer('ffn_projection_mask', torch.empty([144,144 * (num_cnn + 1)]))
            
        self.ffn = nn.Linear(self.ffn_projection_mask.shape[1], self.ffn_projection_mask.shape[0], bias=None)
        
        self.loss_function = None
        self.train_phase = 0
        self.wing_loss_width=wing_loss_width
        

    def forward(self, x, targets, multicrop = None, landmarks = None):
        if self.training:
            self.loss_function = WingLoss(width = self.wing_loss_width)
        else:
            self.loss_function = nn.MSELoss()

        if self.train_phase == 0:
            outputs, losses = self.ensemble(x, targets)
            final_loss = torch.sum(torch.stack(losses, dim=0), dim=0)

            return outputs, final_loss, final_loss/self.ensemble.num_projectors

        elif self.train_phase == 1:
            
            with torch.no_grad():
                # raw_landmarks = self.ensemble.predict(x)
                raw_landmarks = landmarks
                raw_loss = self.loss_function(raw_landmarks, targets)

            try:
                correction = self.cnn_ensemble(multicrop, catenate=False)
                # correction = self.cnn_focusing(multicrop) + self.cnn_focusing2(multicrop) + self.cnn_focusing3(multicrop) + self.cnn_focusing4(multicrop)
            except:
                correction = self.transformer_focusing(multicrop)
            output = raw_landmarks + correction
            final_loss = self.loss_function(output, targets)
            return output, final_loss, raw_loss

        elif self.train_phase >= 2:
            
            raw_landmarks = landmarks
            
            with torch.no_grad():
                # raw_landmarks = self.ensemble.predict(x)
                raw_loss = self.loss_function(raw_landmarks, targets)
                try:
                    correction = self.cnn_ensemble(multicrop, catenate=False)
                except:
                    correction = self.transformer_focusing(multicrop)
            
            # ffn_input = torch.cat((raw_landmarks, correction), dim = 1)
            
            ffn_input = raw_landmarks + correction
            output = self.ffn(ffn_input)
            # output = F.linear(ffn_input, self.ffn.weight*self.ffn_projection_mask, bias=None) # + raw_landmarks
            
            final_loss = self.loss_function(output, targets)

            return output, final_loss, raw_loss    
    
    @torch.no_grad()
    def predict(self, image, face_detail = False, pixels = True, precrop = False, vertical_flip = True):
        
        # Preprocessing pathway
        if precrop:
            image, xmin, ymin, xmax, ymax = crop_face_only(image)
        
        if BOTH_MODELS:
            input_landmarks = np.concatenate((MediaPipe_model(image), LBF_model(image)), axis = 0)
        else:
            input_landmarks = MediaPipe_model(image)
        
        # Facedataset pathway    
        input_landmarks = torch.from_numpy(input_landmarks).float()
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
            multicrop = normalize_multicrop(make_landmark_crops(raw_landmarks, subimage, self.crop_size))
            correction = self.cnn_ensemble(multicrop[None,:,:,:], catenate=False)

            output = raw_landmarks + correction
        
        elif self.train_phase == 2:
            raw_landmarks = self.ensemble.predict(relative_landmarks)
            multicrop = normalize_multicrop(make_landmark_crops(raw_landmarks, subimage, self.crop_size))
            correction = self.cnn_ensemble(multicrop[None,:,:,:])
            
            # ffn_input = torch.cat((raw_landmarks, correction), dim = 1)
            ffn_input = raw_landmarks + correction
            output = F.linear(ffn_input, self.ffn.weight*self.ffn_projection_mask, bias=None) # + raw_landmarks
            
            output = self.ffn(ffn_input)
        
        output = output.reshape(-1,2).detach()

        abs_output = get_absolute_positions(rotate_landmarks(-angle, output, subimage.shape), centroid, size_measure)    
        abs_output = abs_output.cpu().detach().numpy()
        output = output.cpu().numpy()
        
        if vertical_flip:
            output[:,1] = 1 - output[:,1]
            abs_output[:,1] = 1 - abs_output[:,1]
        
        # Pixel dimension
        if pixels:
            output = np.multiply(output, np.array([subimage.shape[1], subimage.shape[0]]).reshape(1,1,2)).astype(np.int32)
            abs_output = np.multiply(abs_output, np.array([image.shape[1], image.shape[0]]).reshape(1,1,2)).astype(np.int32)
        
        if face_detail:
            return output.squeeze(), relative_landmarks, subimage
        
        else:
            return abs_output.squeeze(), input_landmarks, image