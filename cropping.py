import numpy as np
import torch
import cv2
import imagesize
import mediapipe as mp
from facelandmarks.config import DEVICE, STANDARD_IMAGE_WIDTH 


def crop_around_centroid(image, centroid, size_measure):
   
    subimage_center = torch.mul(centroid, torch.tensor([image.shape[1], image.shape[0]]).to(DEVICE))
    subimage_size = torch.mul(size_measure, torch.tensor([image.shape[1], image.shape[0]]).to(DEVICE))

    subimage_margins = torch.cat([-1 * torch.squeeze(subimage_center - subimage_size), torch.squeeze(subimage_center + subimage_size)])
    image_margins = torch.tensor([0,0,image.shape[1], image.shape[0]]).to(DEVICE)

    cropping = torch.max(torch.ones(4).to(DEVICE),image_margins - subimage_margins).type(torch.int)
    padding = torch.abs(torch.min(torch.zeros(4).to(DEVICE),image_margins - subimage_margins)).type(torch.int)

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
            size_coef = 2
            
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

def get_image_edges(image, threshold1=50, threshold2=150):
    img_blur = cv2.GaussianBlur(image, (3,3), 5) 
    edges = cv2.Canny(image=img_blur, threshold1=threshold1, threshold2=threshold2) 
    return edges

def get_subimage_shape(image_path, size_measure):
    try:
      width, height = imagesize.get(image_path)
    except:
      image_path = '/'.join(image_path.split('/')[:-1]) + '/' + image_path.split('/')[-1].split('.')[0] + '.' + image_path.split('/')[-1].split('.')[1].lower()
      width, height = imagesize.get(image_path)
    subimage_size = 2*torch.mul(size_measure, torch.tensor([width, height]).to(DEVICE)).squeeze()
    new_width = STANDARD_IMAGE_WIDTH
    scale = new_width / subimage_size[0]
    new_height = int(scale * subimage_size[1])
    # new_height = 600
    return torch.tensor([new_height, new_width])
    
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
    sub_image = image[crop_indices[:,:,1,:], crop_indices[:,:,0,:], :]

    # Final shape (3 for RGB * num_landmarks, x_crop_size, y_crop_size)
    # cnn in torch requires channels first
    multicrop = sub_image.reshape(crop_size, crop_size, -1).permute(2,0,1).type(torch.float).to(DEVICE)

    return multicrop

def normalize_multicrop(multicrop):
    if multicrop.shape[0] == 72:
        unstacked_multicrop = torch.unflatten(multicrop, 0, (-1, 1))  # shape (72,1,height, width)
    else:
        unstacked_multicrop = torch.unflatten(multicrop, 0, (-1, 3))   # shape (72,3,height, width)
    means = []
    stds = []
    for channel in range(unstacked_multicrop.shape[1]):
        means.append(torch.mean(unstacked_multicrop[:,channel,...], dim=(0,1,2), keepdim=True))
        stds.append(torch.std(unstacked_multicrop[:,channel,...], dim=(0,1,2), keepdim=True))
    mean = torch.cat(means, dim=1).unsqueeze(-1)
    std = torch.cat(stds, dim=1).unsqueeze(-1)
    
    normalized_multicrop = torch.clamp(torch.div(torch.sub(unstacked_multicrop, mean), 3 * std) + 0.5, 0, 1)

    return torch.flatten(normalized_multicrop, start_dim=0, end_dim=1)

def template_matching(multicrop, avg_template, template_method, crop_as_template = False):
        
    crops = np.split(multicrop.numpy(), indices_or_sections=multicrop.shape[-3], axis=-3)
    templates = np.split(avg_template.numpy(), indices_or_sections=avg_template.shape[-3], axis=-3)
    matches = np.empty([1,0])

    for crop, template in zip(crops, templates):
        if crop_as_template:
            match = cv2.matchTemplate(template.squeeze(), crop.squeeze(), template_method) # 2D
        else:
            match = cv2.matchTemplate(crop.squeeze(), template.squeeze(), template_method) # 2D
        
        # TODO: ? Return some better format than concatenated vector?
        # result shape (1,height*width*RGB*72)
        matches = np.concatenate([matches, match.reshape(1,-1)], axis= 1)

    # TODO: Remove magic constant
    return torch.from_numpy(0.001 * matches/255).squeeze().type(torch.float32)