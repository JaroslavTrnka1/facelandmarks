from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from landmarks_utils import *
from cropping import *
from config import *



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

# class CNNFocusing(nn.Module):
#     def __init__(self, crop_size, monolayer = True, hidden_per_lmark=10):
#         super().__init__()
#         # TODO: general approach to number of cnn layer and convolutions
#         self.conv1 = nn.Conv2d(216, 216, 3, stride = 1, groups = 72, padding = 'same', padding_mode = 'replicate')
#         self.conv2 = nn.Conv2d(216, 432, 3, stride = 1, groups = 72, padding = 'same', padding_mode = 'replicate')
#         #self.conv3 = nn.Conv2d(72, 72, 3, stride = 1, groups = 72, padding = 'same', padding_mode = 'replicate')

#         pool = 2
#         self.pool1 = nn.MaxPool2d((pool,pool))
#         #self.pool2 = nn.MaxPool2d(3)

#         ch_per_lmark = self.conv2.weight.shape[0] // 72
        
#         if monolayer:
#             hidden_per_lmark = 2
            
#         hidden_dim = int((crop_size/pool)**2 * ch_per_lmark)
#         mask_diag = torch.diag(torch.ones(72))
#         linear1_mask = mask_diag.repeat_interleave(hidden_per_lmark, dim = 1).repeat_interleave(hidden_dim, dim = 0)
#         # linear2_mask = mask_diag.repeat_interleave(2, dim = 1).repeat_interleave(hidden_per_lmark, dim = 0)

#         self.register_buffer("mask1", linear1_mask)
#         # self.register_buffer("mask2", linear2_mask)

#         self.linear1 = nn.Linear(linear1_mask.shape[0], linear1_mask.shape[1], bias = False)
#         # self.linear2 = nn.Linear(linear2_mask.shape[0], linear2_mask.shape[1])

#         self.crop_size = crop_size
    
#     def forward(self, x):

#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = F.relu(self.conv2(x))
#         x = torch.flatten(x, 1)

#         self.linear1.weight.data.mul_(self.mask1.permute(1,0))
#         output = self.linear1(x)

#         # self.linear2.weight.data.mul_(self.mask2.permute(1,0))
#         # output = self.linear2(output)

#         return output

class CNNFocusing(nn.Module):
    def __init__(self, kernel_sizes, activations = True, pooling = True, batch_norm = False, crop_size = 46, start_out_channels = 8):
        super().__init__()
        
        self.cnn = self.initialize_cnn(kernel_sizes, activations, pooling, start_out_channels=start_out_channels, batch_norm=batch_norm)
        cnn_output_size = self.calculate_output_size(crop_size, self.cnn)
        hidden_per_lmark = 64
        mask_diag = torch.diag(torch.ones(72))
        linear1_mask = mask_diag.repeat_interleave(hidden_per_lmark, dim = 1).repeat_interleave(int(cnn_output_size), dim = 0)
        linear2_mask = mask_diag.repeat_interleave(2, dim = 1).repeat_interleave(hidden_per_lmark, dim = 0)

        self.register_buffer("mask1", linear1_mask)
        self.register_buffer("mask2", linear2_mask)
        
        self.linear1 = nn.Linear(linear1_mask.shape[0], linear1_mask.shape[1], bias = False)
        self.linear2 = nn.Linear(linear2_mask.shape[0], linear2_mask.shape[1])
    
    def forward(self, x):
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
            

# class TemplateMatchProjection(nn.Module):
#     def __init__(self, template_mask, bias = None):
#         super().__init__()

#         # should be (height*width*RGB*72)
#         input_dim = template_mask.shape[1]
#         self.linear = nn.Linear(input_dim, 144, bias = bias)
#         self.register_buffer('mask', template_mask)
#         self.bias = bias

#     def forward(self, x):
#         output = F.linear(x, self.linear.weight*self.mask, bias=self.bias)
#         return output
    
# class PretrainedResNet(nn.Module):
#     def __init__(self, num_landmarks, pretrained_resnet=True):
#         super().__init__()

#         self.backbone = models.resnet34(pretrained=pretrained_resnet)
        
#         # Remove the fully connected layers at the end of ResNet
#         self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

#         # Additional convolutional layers if needed
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(in_channels=..., out_channels=..., kernel_size=...),
#             nn.ReLU(),
#             # Add more convolutional layers if necessary
#         )
        
#         # Flatten the output
#         self.flatten = nn.Flatten()

#         # Fully connected layers for landmark coordinates
#         self.fc_landmarks = nn.Sequential(
#             nn.Linear(in_features=..., out_features=..., bias=True),
#             nn.ReLU(),
#             nn.Linear(in_features=..., out_features=num_landmarks * 2, bias=True)
#             # num_landmarks * 2 because each landmark has x and y coordinates
#         )

#     def forward(self, x):
#         # Forward pass through the network
#         x = self.backbone(x)
#         x = self.conv_layers(x)
#         x = self.flatten(x)
#         landmarks = self.fc_landmarks(x)
#         return landmarks   

class EnsembleCNN(nn.Module):
    def __init__(self, num_cnn, kernel_sizes, activations, pooling, batch_norm, crop_size, start_out_channels):
        super().__init__()
        
        self.ensembleCNN = nn.ModuleList()
        self.num_cnn = num_cnn
        
        for i in range(num_cnn):
            cnn_focusing = CNNFocusing(
                kernel_sizes=kernel_sizes,
                activations=activations,
                pooling=pooling[i],
                batch_norm=batch_norm,
                crop_size = crop_size,
                start_out_channels=start_out_channels
                )
            self.ensembleCNN.append(cnn_focusing)
    
    def forward(self, x, catenate = True):
        output = self.ensembleCNN[0](x)
        for cnn in self.ensembleCNN[1:]:
            if catenate:
                output = torch.cat([output, cnn(x)], dim = 1)
            else:
                output += cnn(x)
        return output
    
    @torch.no_grad()
    def predict(self, x, catenate = True):
        # if x.shape[0] == 216:
        #     x = torch.unsqueeze(x, dim = 0)
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
                 template_mode = False,
                 avg_template = None,
                 template_method = None,
                 crop_as_template = False,
                 batch_norm = False,
                 num_cnn = 4
                 ):
        
        super().__init__()

        if BOTH_MODELS:
            input_dim = 1092
        else:
            input_dim = 956
            
        self.ensemble = EnsembleProjection(input_dim, 144, projectors, projection_mask)
        self.template_mode = template_mode

        if template_mode:
            self.templ_match_projection = TemplateMatchProjection(template_mask=template_mask)
        else:
            cnn_pooling = []
            while len(cnn_pooling) < num_cnn:
                cnn_pooling.extend(pooling)
                cnn_pooling = cnn_pooling[:num_cnn]
            self.cnn_ensemble = EnsembleCNN(num_cnn=num_cnn,
                                            kernel_sizes=kernel_sizes,
                                            activations=activations,
                                            pooling=cnn_pooling,
                                            batch_norm=batch_norm,
                                            crop_size = crop_size,
                                            start_out_channels=start_out_channels)
            self.cnn_ensemble2 = EnsembleCNN(num_cnn=num_cnn,
                                            kernel_sizes=kernel_sizes,
                                            activations=activations,
                                            pooling=cnn_pooling,
                                            batch_norm=batch_norm,
                                            crop_size = crop_size,
                                            start_out_channels=start_out_channels)
            
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
        
        self.loss_function = nn.MSELoss()
        self.train_phase = 0
        
        if self.template_mode: 
            self.template = avg_template
            self.template_method = eval(str(template_method))
            self.crop_as_template = crop_as_template


    def forward(self, x, targets, multicrop = None, landmarks = None):

        if self.train_phase == 0:
            outputs, losses = self.ensemble(x, targets)
            final_loss = torch.sum(torch.stack(losses, dim=0), dim=0)

            return outputs, final_loss, final_loss/self.ensemble.num_projectors

        elif self.train_phase == 1:
            
            with torch.no_grad():
                # raw_landmarks = self.ensemble.predict(x)
                raw_landmarks = landmarks
                raw_loss = self.loss_function(raw_landmarks, targets)

            if self.template_mode:
                correction = self.templ_match_projection(multicrop)
            else:
                correction = self.cnn_ensemble(multicrop, catenate=False)
                # correction = self.cnn_focusing(multicrop) + self.cnn_focusing2(multicrop) + self.cnn_focusing3(multicrop) + self.cnn_focusing4(multicrop)
            
            output = raw_landmarks + correction
            final_loss = self.loss_function(output, targets)
            return output, final_loss, raw_loss

        elif self.train_phase >= 2:
            
            raw_landmarks = landmarks
            
            with torch.no_grad():
                # raw_landmarks = self.ensemble.predict(x)
                raw_loss = self.loss_function(raw_landmarks, targets)
                # correction = self.cnn_ensemble(multicrop)
                
            if self.template_mode:
                correction = self.templ_match_projection(multicrop)
                
            correction = self.cnn_ensemble2(multicrop, catenate = False)    
            output = landmarks + correction
            
            # ffn_input = torch.cat((raw_landmarks, correction), dim = 1)
            
            # ffn_input = raw_landmarks + correction
            # output = self.ffn(ffn_input)
            # output = F.linear(ffn_input, self.ffn.weight*self.ffn_projection_mask, bias=None) # + raw_landmarks
            
            final_loss = self.loss_function(output, targets)

            return output, final_loss, raw_loss    
    
    @torch.no_grad()
    def predict(self, image, face_detail = True, gray = False):
        
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
        
        if gray:
            subimage = cv2.cvtColor(subimage, cv2.COLOR_BGR2GRAY)[:,:,None]
        
        # Model pathway
        if self.train_phase == 0:
            output = self.ensemble.predict(relative_landmarks)
 
        elif self.train_phase == 1:
            raw_landmarks = self.ensemble.predict(relative_landmarks)
            
            multicrop = normalize_multicrop(make_landmark_crops(raw_landmarks, subimage, self.crop_size))
            # img_edges = get_image_edges(subimage)[:,:,None]
            # multicrop = make_landmark_crops(raw_landmarks, img_edges, self.crop_size)
            if self.template_mode:
                multicrop = template_matching(multicrop, self.template, self.template_method, crop_as_template=self.crop_as_template)
                correction = self.templ_match_projection(multicrop)
            else:
                correction = self.cnn_ensemble(multicrop[None,:,:,:], catenate=False)

            output = raw_landmarks + correction
        
        elif self.train_phase == 2:
            raw_landmarks = self.ensemble.predict(relative_landmarks)
            
            multicrop = normalize_multicrop(make_landmark_crops(raw_landmarks, subimage, self.crop_size))
            # img_edges = get_image_edges(subimage)[:,:,None]
            # multicrop = make_landmark_crops(raw_landmarks, img_edges, self.crop_size)

            if self.template_mode:
                multicrop = template_matching(multicrop, self.template, self.template_method, crop_as_template=self.crop_as_template)
                correction = self.templ_match_projection(multicrop)
            else:
                correction = self.cnn_ensemble(multicrop[None,:,:,:])
            
            ffn_input = torch.cat((raw_landmarks, correction), dim = 1)
            # ffn_input = raw_landmarks + correction
            # output = F.linear(ffn_input, self.ffn.weight*self.ffn_projection_mask, bias=None) # + raw_landmarks
            
            output = self.ffn(ffn_input)
        
        output = output.reshape(-1,2).detach()
        #output = output.unflatten(-1, (-1, 2)).cpu().detach()
        # output shape: (1, 72, 2) - důležité pro případný batching

        abs_output = get_absolute_positions(rotate_landmarks(-angle, output, subimage.shape), centroid, size_measure)    
        abs_output = abs_output.cpu().detach().numpy()
        output = output.cpu().numpy()
        
        # Pixel dimension
        # pixel_multiplyer = np.array([subimage.shape[1], subimage.shape[0]]).reshape(1,1,2)
        # output = np.multiply(output, pixel_multiplyer).astype(np.int32)
        # abs_output = np.multiply(abs_output, pixel_multiplyer).astype(np.int32)
        
        if face_detail:
            return output, relative_landmarks, subimage
        
        else:
            return abs_output, input_landmarks, image