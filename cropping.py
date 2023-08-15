import numpy as np
import torch
import cv2
import mediapipe as mp

from config import *


def crop_face_only(image):
    # Initialize the face detection module and process the image
    face_detection = mp.solutions.face_detection.FaceDetection()
    results = face_detection.process(image)
    
    # Check if any faces were detected
    if results.detections:
        
        for detection in results.detections:
            # Extract the bounding box coordinates
            bbox = detection.location_data.relative_bounding_box
            image_height, image_width, _ = image.shape
            
            # accomodation of the box size
            size_coef = 1.7
            
            width = int(bbox.width * image_width * size_coef)
            height = int(bbox.height * image_height * size_coef)
            
            xmin = int(bbox.xmin * image_width - (width/size_coef) * (size_coef - 1)/2)
            ymin = int(bbox.ymin * image_height - (height/size_coef) * (size_coef - 1)/1.3)
            
            xmax = min(xmin + width, image_width)
            ymax = min(ymin + height, image_height)
            
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            
            subimage = np.ascontiguousarray(image[ymin:ymax, xmin:xmax, :])
    
        return subimage, xmin, ymin, xmax, ymax
    
    else:
        return None 
 
 
def standard_face_size(image):
    new_width = 500
    scale = new_width / image.shape[1]
    new_height = int(scale * image.shape[0])
    image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_AREA)
    return image

    
@torch.no_grad()
def make_landmark_crops(raw_landmarks, image, crop_size):

    # Scaling from (0,1) to pixel scale and transposing landmarks
    raw_landmarks_pix = torch.mul(raw_landmarks.reshape(-1,2), torch.tensor([image.shape[1], image.shape[0]]).to(DEVICE)).permute(1,0)

    # Preparing index matrices of all crops

    crop_range = torch.arange(-crop_size // 2, crop_size // 2)

    # shape (30,30,2) --> one layer of horizontal indices from -15 to 14, second the same verical
    crop_matrix = torch.stack([crop_range.tile((crop_size,1)), crop_range[:, None].tile((1,crop_size))], dim = 2).to(DEVICE)

    # shape: (x_coor_matrix horizontal, y_coor_matrix vertical, 2, num_landmarks)
    crop_indices = (raw_landmarks_pix[None, None,:,:] + crop_matrix[:,:,:,None]).type(torch.LongTensor) # float to int for indices

    image = torch.tensor(image).to(DEVICE)
    # Cropping image around raw landmarks
    sub_image = (image[crop_indices[:,:,1,:], crop_indices[:,:,0,:], :]).clone().detach()

    # Final shape (3 for RGB * num_landmarks, x_crop_size, y_crop_size)
    multicrop = sub_image.reshape(crop_size, crop_size, -1).permute(2,0,1).type(torch.float).to(DEVICE)

    return multicrop