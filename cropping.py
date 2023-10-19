import numpy as np
import torch
import cv2
import mediapipe as mp
import imagesize
from config import *


def crop_around_centroid(image, centroid, size_measure):
    
    subimage_center = torch.mul(centroid, torch.tensor([image.shape[1], image.shape[0]]))
    subimage_size = torch.mul(size_measure, torch.tensor([image.shape[1], image.shape[0]]))

    subimage_margins = torch.cat([-1 * torch.squeeze(subimage_center - subimage_size), torch.squeeze(subimage_center + subimage_size)])
    image_margins = torch.tensor([0,0,image.shape[1], image.shape[0]])

    cropping = torch.max(torch.ones(4),image_margins - subimage_margins).type(torch.int)
    padding = torch.abs(torch.min(torch.zeros(4),image_margins - subimage_margins)).type(torch.int)

    padded_img = cv2.copyMakeBorder(image,padding[1].item(),padding[3].item(),padding[0].item(),padding[2].item(),cv2.BORDER_REPLICATE)
    subimage = np.ascontiguousarray(padded_img[cropping[1]:-cropping[3], cropping[0]:-cropping[2],:])
    
    return subimage


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
    new_width = STANDARD_IMAGE_WIDTH
    scale = new_width / image.shape[1]
    new_height = int(scale * image.shape[0])
    image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_AREA)
    return image

def rotate_image(angle_deg, image):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated_image

def get_subimage_shape(image_path, size_measure):
    width, height = imagesize.get(image_path)
    subimage_size = 2*torch.mul(size_measure, torch.tensor([width, height])).squeeze()
    new_width = STANDARD_IMAGE_WIDTH
    scale = new_width / subimage_size[0]
    new_height = int(scale * subimage_size[1])
    return torch.tensor([new_height, new_width])
    
@torch.no_grad()
def make_landmark_crops(raw_landmarks, image, crop_size):
    try:

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
    
    except:
        print(raw_landmarks, image.shape, crop_size)

    return multicrop