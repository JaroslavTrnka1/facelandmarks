import numpy as np
import torch
import cv2
import imagesize
import mediapipe as mp
from facelandmarks.config import DEVICE, STANDARD_IMAGE_WIDTH 


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
    # try:
    width, height = imagesize.get(image_path)
    # except:
    #   image_path = '/'.join(image_path.split('/')[:-1]) + '/' + image_path.split('/')[-1].split('.')[0] + '.' + image_path.split('/')[-1].split('.')[1].lower()
    #   width, height = imagesize.get(image_path)
    subimage_size = 2*torch.mul(size_measure, torch.tensor([width, height])).squeeze()
    new_width = STANDARD_IMAGE_WIDTH
    scale = new_width / subimage_size[0]
    new_height = int(scale * subimage_size[1])
    # new_height = 600
    return torch.tensor([new_height, new_width])
    
@torch.no_grad()
def make_landmark_crops(raw_landmarks, image, crop_size):
    
    # Scaling from (0,1) to pixel scale and transposing landmarks
    raw_landmarks_pix = torch.mul(raw_landmarks.reshape(-1,2), torch.tensor([image.shape[1], image.shape[0]])).permute(1,0)
    
    # Preparing index matrices of all crops
    crop_range = torch.arange(-crop_size // 2, crop_size // 2)

    # shape (30,30,2) --> one layer of horizontal indices from -15 to 14, second the same verical
    crop_matrix = torch.stack([crop_range.tile((crop_size,1)), crop_range[:, None].tile((1,crop_size))], dim = 2)

    # shape: (x_coor_matrix horizontal, y_coor_matrix vertical, 2, num_landmarks)
    crop_indices = (raw_landmarks_pix[None, None,:,:] + crop_matrix[:,:,:,None]).type(torch.LongTensor) # float to int for indices

    image = torch.tensor(image)
    
    # Cropping image around raw landmarks
    sub_image = image[crop_indices[:,:,1,:], crop_indices[:,:,0,:], :]

    # Final shape (3 for RGB * num_landmarks, x_crop_size, y_crop_size)
    # cnn in torch requires channels first
    multicrop = sub_image.reshape(crop_size, crop_size, -1).permute(2,0,1).type(torch.float)

    return multicrop

@torch.no_grad()
def get_landmark_crop_position_unbatched(raw_landmarks, true_landmarks, image, crop_size, return_float = True):
    
    # Scaling from (0,1) to pixel scale and transposing landmarks
    raw_landmarks_pix = torch.mul(raw_landmarks.reshape(-1,2), torch.tensor([image.shape[1], image.shape[0]]))
    true_landmarks_pix = torch.mul(true_landmarks.reshape(-1,2), torch.tensor([image.shape[1], image.shape[0]]))
    
    differences_pix = raw_landmarks_pix - true_landmarks_pix
    
    if not return_float:
        return differences_pix.type(torch.int16) + int(crop_size//2)
    
    else:
        return torch.div(differences_pix, crop_size).type(torch.int16) + 0.5
    
    
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


def create_heatmaps_from_crop_landmarks(landmarks_coords, crop_size, sigma=None):
    """
    Create ground truth heatmaps for multiple landmarks.
    
    Parameters:
    - landmarks_coords: tensor of shape (num_landmarks, 2) containing (x, y) coordinates of landmarks.
    - heatmap_size: int
    - sigma: standard deviation for the Gaussian (controls spread of the heatmaps).
    
    Returns:
    - A 3D tensor of shape (num_landmarks, height, width) representing the heatmaps.
    """
    sigma = crop_size//30
    num_landmarks = landmarks_coords.shape[0]
    
    # Initialize an empty tensor to hold the heatmaps for all landmarks
    heatmaps = torch.zeros((num_landmarks, crop_size, crop_size))
    
    # Create a grid of (x, y) coordinates representing each pixel location in the heatmap
    xx, yy = torch.meshgrid(torch.arange(crop_size), torch.arange(crop_size))
    xx = xx.T  # Transpose to match the (height, width) format
    yy = yy.T
    
    for i, (x, y) in enumerate(landmarks_coords):
        # Calculate the 2D Gaussian distribution centered at (x, y) for each landmark
        gaussian = torch.exp(- ((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        # Normalize the Gaussian so that the peak value is 1 (optional)
        gaussian /= gaussian.max()
        # Store the heatmap for this landmark
        heatmaps[i] = gaussian
    return heatmaps


def create_true_heatmaps(raw_landmarks, true_landmarks, image, crop_size):
    crop_positions = get_landmark_crop_position_unbatched(raw_landmarks, true_landmarks, image, crop_size, return_float=False)   # (num_landmarks, 2)
    heatmaps = create_heatmaps_from_crop_landmarks(crop_positions, crop_size)
    return heatmaps

# Batched function
def get_crop_center_preds_from_heatmap_batch(heatmap):

    # maxm = torch.nn.MaxPool2d(3, 1, 1)(heatmap)
    # maxm = torch.eq(maxm, heatmap).float()
    # heatmap = heatmap * maxm
    
    # h = heatmap.size()[2]
    # w = heatmap.size()[3]
    # heatmap = heatmap.view(heatmap.size()[0], heatmap.size()[1], -1)
    # val_k, ind = heatmap.topk(1, dim=2)

    # x = ind % w
    # y = (ind / w).long()
    # ind_k = torch.stack((x, y), dim=3)
    
    max, idx = torch.max(
        heatmap.view(heatmap.size(0), heatmap.size(1), heatmap.size(2) * heatmap.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % heatmap.size(3))# + 1)
    preds[..., 1].add_(-1).div_(heatmap.size(2)).floor_()#.add_(1)

    # for i in range(preds.size(0)):
    #     for j in range(preds.size(1)):
    #         hm_ = heatmap[i, j, :]
    #         pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
    #         if pX > 0 and pX < 63 and pY > 0 and pY < 63:
    #             diff = torch.FloatTensor(
    #                 [hm_[pY, pX + 1] - hm_[pY, pX - 1],
    #                  hm_[pY + 1, pX] - hm_[pY - 1, pX]])
    #             preds[i, j].add_(diff.sign_().mul_(.25))

    # preds.add_(1)
    preds.sub_(torch.tensor(heatmap.size()[2:]) // 2)
    return preds

# Batched function
def get_image_correction_from_heatmap(heatmaps, image_shapes):
    # heatmaps of shape (batch, 72, width, heigt) or (batch, stack, 72, width, height)
    if len(heatmaps.shape) == 5:
        heatmaps = heatmaps[:,-1,...]
    # image_shapes of shape (batch, 2)
    # crop predictions of shape (batch, num_landmarks, 2)
    image_shapes = image_shapes.unsqueeze(-2)
    crop_predictions = get_crop_center_preds_from_heatmap_batch(heatmaps)
    image_correction = torch.div(crop_predictions, image_shapes)
    return image_correction.reshape(-1,144)


