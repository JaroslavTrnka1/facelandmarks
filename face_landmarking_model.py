from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from facelandmarks.landmarks_utils import *
from facelandmarks.cropping import create_true_heatmaps\
    ,crop_face_only, crop_around_centroid, standard_face_size\
        , rotate_image, normalize_multicrop, make_landmark_crops\
            ,get_image_correction_from_heatmap
from facelandmarks.config import *
from facelandmarks.transformer_focusing import GroupedTransformer
from facelandmarks.stacked_hourglass import Hourglass, StackedHourglass




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
    
    
class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

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
    def __init__(self, kernel_sizes, activations = True, pooling = True, padding = 0, batch_norm = False, crop_size = 46, start_out_channels = 8,
                 hidden_per_lmark = 64, cnn_dropout=False):
        super().__init__()
        
        self.cnn = self.initialize_cnn(kernel_sizes, activations, pooling, padding, start_out_channels=start_out_channels, batch_norm=batch_norm)
        self.cnn_output_size, self.out_channels_per_lmark = self.calculate_output_size(crop_size, self.cnn)
        self.crop_size = crop_size
        self.cnn_dropout = cnn_dropout
        
        if hidden_per_lmark == 0:
            self.linears = nn.ModuleList( [
                    nn.Linear(self.cnn_output_size, 2, bias = True)
            for i in range(72)] )
        else:
            self.linears = nn.ModuleList( [
                nn.Sequential(
                    nn.Linear(self.cnn_output_size, hidden_per_lmark, bias = False),
                    nn.Linear(hidden_per_lmark, 2, bias = True)
                )
            for i in range(72)] )
        
        # mask_diag = torch.diag(torch.ones(72))
        # if hidden_per_lmark == 0:
        #     self.two_layers = False
        #     linear0_mask = mask_diag.repeat_interleave(2, dim = 1).repeat_interleave(int(cnn_output_size), dim = 0)
        #     self.register_buffer("mask0", linear0_mask)
        #     self.linear0 = nn.Linear(linear0_mask.shape[0], linear0_mask.shape[1], bias = True)
        # else:
        #     self.two_layers = True
        #     linear1_mask = mask_diag.repeat_interleave(hidden_per_lmark, dim = 1).repeat_interleave(int(cnn_output_size), dim = 0)
        #     linear2_mask = mask_diag.repeat_interleave(2, dim = 1).repeat_interleave(hidden_per_lmark, dim = 0)
        #     self.register_buffer("mask1", linear1_mask)
        #     self.register_buffer("mask2", linear2_mask)        
        #     self.linear1 = nn.Linear(linear1_mask.shape[0], linear1_mask.shape[1], bias = False)
        #     self.linear2 = nn.Linear(linear2_mask.shape[0], linear2_mask.shape[1], bias = True)
        
        if self.cnn_dropout:
            self.dropout = nn.Dropout(p = 0.1)
    
    def forward(self, x):
        if x.shape[-1] != self.crop_size:
            new_crop = x.shape[-1]
            x = x[...,int((new_crop - self.crop_size)/2): int((new_crop + self.crop_size)/2), int((new_crop - self.crop_size)/2): int((new_crop + self.crop_size)/2)]
        x = self.cnn(x)
        x = torch.flatten(x, -2)
        landmark_results = []
        for i in range(72):
            landmark_cnn_output = torch.flatten(x[:, i * self.out_channels_per_lmark : (i + 1) * self.out_channels_per_lmark, :], -2)
            landmark_results.append(self.linears[i](landmark_cnn_output))
   
        output = torch.cat(landmark_results, dim = 1)
        # x = torch.flatten(x, -3)
        # if self.two_layers:
        #     self.linear1.weight.data.mul_(self.mask1.permute(1,0))
        #     output = self.linear1(x)
        #     if self.cnn_dropout:
        #         self.dropout(output)
        #     self.linear2.weight.data.mul_(self.mask2.permute(1,0))
        #     output = self.linear2(output)
        # else:
        #     self.linear0.weight.data.mul_(self.mask0.permute(1,0))
        #     output = self.linear0(x)
        return output
    
        
    def initialize_cnn(self, kernel_sizes: list, activations = False, pooling = True, padding = 0, batch_norm = False, start_out_channels = 8):
        if not isinstance(pooling, list):
            pooling = [pooling] * len(kernel_sizes)
        if not isinstance(activations, list):
            activations = [activations] * len(kernel_sizes)
        cnn_layers = []
        in_channels = 3 
        out_channels = start_out_channels
        for kernel_size, activation, pool in zip(kernel_sizes, activations, pooling):
            if kernel_size == 1:
                cnn_layers.append(nn.Conv2d(in_channels * 72, 1 * 72, kernel_size, padding=0, groups=72))
            else:
                cnn_layers.append(nn.Conv2d(in_channels * 72, out_channels * 72, kernel_size, padding=padding, groups=72))
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
        
        input_size = out_channels * input_size**2 / 72
        return int(input_size), int(out_channels / 72)
            

class EnsembleCNN(nn.Module):
    def __init__(self, num_cnn, kernel_sizes, activations, pooling, padding, batch_norm, crop_size, start_out_channels, hidden_per_lmark, mixture = False,
                 cnn_dropout=False):
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
                padding=padding,
                batch_norm=batch_norm,
                crop_size = crop_size[i],
                start_out_channels=start_out_channels,
                hidden_per_lmark = hidden_per_lmark,
                cnn_dropout=cnn_dropout
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
                 padding = 0,
                 crop_size = 46,
                 start_out_channels=8,
                 ffn_projection_mask = None,
                 batch_norm = False,
                 num_cnn = 4,
                 wing_loss_width=5,
                 hidden_per_lmark=64,
                 mixture = None,
                 cnn_dropout = False,
                 top_head = 'cnn'
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
            
        self.heatmaps = False
        if FOCUSING == 'CNN':
            self.focusing = EnsembleCNN(num_cnn=num_cnn,
                                            kernel_sizes=kernel_sizes,
                                            activations=activations,
                                            pooling=cnn_pooling,
                                            padding=padding,
                                            batch_norm=batch_norm,
                                            crop_size = crop_size,
                                            start_out_channels=start_out_channels,
                                            hidden_per_lmark=hidden_per_lmark,
                                            mixture=mixture,
                                            cnn_dropout=cnn_dropout)
            
        elif FOCUSING == 'heatmaps':
            self.heatmaps = True
            self.focusing = StackedHourglass(4, start_out_channels, 1, num_landmarks=72, hourglass_depth=4)
        else:
            self.focusing = GroupedTransformer(crop_size=crop_size,
                                                    num_crops=72,
                                                    patch_size=5,
                                                    embed_dim=64,
                                                    num_heads=4,
                                                    num_blocks=4)
        

        if isinstance(crop_size, list):
            self.crop_size = max(crop_size)
        else:
            self.crop_size = crop_size
            
        if top_head == 'ffn-simple':    
            self.ffn = nn.Sequential(
              nn.Linear(144, 288 * 2, bias=False),
              nn.Tanh(),
              nn.Linear(288 * 2, 144, bias=False),
            )    
        elif top_head == 'ffn-catenate':
            if ffn_projection_mask is not None:
                self.register_buffer('ffn_projection_mask', ffn_projection_mask)
            else:
                self.register_buffer('ffn_projection_mask', torch.empty([144,144 * (num_cnn + 1)]))
                
            self.ffn = nn.Linear(self.ffn_projection_mask.shape[1], self.ffn_projection_mask.shape[0], bias=None)
        elif top_head == 'cnn':
            self.ffn = CNNFocusing(
                    kernel_sizes=[3,3,3],
                    activations=False,
                    pooling=False,
                    batch_norm=False,
                    crop_size = 46,
                    start_out_channels=8,
                    hidden_per_lmark = 0,
                    cnn_dropout=False
                    )
        else:
            self.ffn = None
        
        self.top_head = top_head
            
        self.loss_function = None
        self.train_phase = 0
        self.wing_loss_width=wing_loss_width
        

    def forward(self, x, targets, multicrop = None, landmarks = None, heatmaps_true = None, image_sizes = None):
        if self.training:
            if self.heatmaps:
                # self.loss_function = AdaptiveWingLoss()
                self.loss_function = nn.MSELoss()
            else:
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

            if self.heatmaps:
                heatmap_preds_stacked = self.focusing(multicrop)
                
                if self.training:
                    final_loss = self.focusing.calc_loss(heatmap_preds_stacked, heatmaps_true)
                    output = None
                else:
                    correction = get_image_correction_from_heatmap(heatmap_preds_stacked, image_sizes)
                    print(f'\n Correction mean: {correction.mean().item()}')
                    output = raw_landmarks + correction
                    final_loss = self.loss_function(output, targets)
                
                return output, final_loss, raw_loss
                    
            else:                
                correction = self.focusing(multicrop, catenate=False)

                output = raw_landmarks + correction
                final_loss = self.loss_function(output, targets)
                return output, final_loss, raw_loss

        elif self.train_phase >= 2:
            
            raw_landmarks = landmarks
            
            if self.top_head == 'cnn':
                raw_loss = self.loss_function(raw_landmarks, targets)
                correction = self.ffn(multicrop)         
                output = correction + raw_landmarks
            elif self.top_head == 'ffn-simple':
                raw_loss = self.loss_function(raw_landmarks, targets)
                output = self.ffn(raw_landmarks)
            elif self.top_head == 'ffn-catenate':
                raw_loss = self.loss_function(raw_landmarks[:,:144], targets)
                output = F.linear(raw_landmarks, self.ffn.weight*self.ffn_projection_mask, bias=True)
            
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
            
            if self.heatmaps:
                heatmap_preds_stacked = self.focusing(multicrop[None,:,:,:])
                correction = get_image_correction_from_heatmap(heatmap_preds_stacked, torch.tensor([subimage.shape[1], subimage.shape[0]])[None, :,:])
            else:                
                correction = self.focusing(multicrop[None,:,:,:], catenate=False)

            output = raw_landmarks + correction
        
        elif self.train_phase == 2:
            raw_landmarks = self.ensemble.predict(relative_landmarks)
            multicrop = normalize_multicrop(make_landmark_crops(raw_landmarks, subimage, self.crop_size))
            correction = self.focusing(multicrop[None,:,:,:])
            
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