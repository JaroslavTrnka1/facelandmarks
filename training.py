import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances

from landmarks_utils import *
from cropping import *
from config import *

class SparsityLoss(nn.Module):
    def __init__(self, max_non_zero_connections, threshold=0.001):
        super(SparsityLoss, self).__init__()
        self.max_non_zero_connections = max_non_zero_connections
        self.threshold = threshold

    def forward(self, weight_matrix):
        # Calculate the L0 norm approximation using soft thresholding
        soft_threshold = torch.abs(weight_matrix) - self.threshold
        l0_norm = torch.sum(torch.relu(soft_threshold), dim=0)
        
        # Calculate the excess connections beyond the desired limit
        excess_connections = l0_norm - self.max_non_zero_connections
        
        # Define a penalty term to encourage sparsity
        penalty = torch.sum(torch.relu(excess_connections))
        return penalty
    
    
def create_projection_mask(num_parent_landamrks=5):
    preprocessed_inputs = np.load('preprocessed_data/preprocessed_inputs.npz')
    inputs = preprocessed_inputs['x_inp']
    targets = preprocessed_inputs['y_inp']
    n = 20
    cache = None
    for i in range(n):
        face = np.random.randint(inputs.shape[0])
        parent_landmarks = inputs[face,:].reshape(-1,2)
        children_landmarks = targets[face,:].reshape(-1,2)
        similarity_matrix = euclidean_distances(children_landmarks, parent_landmarks).reshape(72,-1,1)
        if cache is None:
            cache=similarity_matrix
        else:
            cache=np.concatenate([cache, similarity_matrix], axis=2)

    avg_sim_matrix = np.mean(cache, axis = 2)
    mask = avg_sim_matrix < np.partition(avg_sim_matrix, num_parent_landamrks, axis=1)[:,num_parent_landamrks:num_parent_landamrks+1]
    
    return torch.from_numpy(mask).repeat_interleave(2, dim = 0).repeat_interleave(2, dim=1)

def prepare_trainers(best_groups, 
                     num_parent_landmarks = 10, 
                     projectors=5, rotate=True, 
                     lr=0.01):
    
    projection_mask = create_projection_mask(num_parent_landamrks=num_parent_landmarks)    
    model = FaceLandmarking(projection_mask, projectors=projectors).to(DEVICE)
    main_dataset = FaceDataset(model, subgroups=best_groups, rotate=rotate)
    main_dataloader = DataLoader(main_dataset, batch_size=100, shuffle=True)

    ensemble_dataset = []
    ensemble_dataloader = []

    print('creating datasets')
    for group in best_groups[:projectors]:
        simple_dataset = FaceDataset(model, subgroups=[group], rotate=rotate)
        ensemble_dataset.append(simple_dataset)
        ensemble_dataloader.append(DataLoader(simple_dataset, batch_size=50, sampler=EnsembleSampler(simple_dataset)))
    
    ensemble_optimizer = torch.optim.Adam(model.ensemble.parameters(), lr = lr)
    cnn_optimizer = torch.optim.Adam(model.cnn_focusing.parameters(), lr = lr)
    ffn_optimizer = torch.optim.Adam(model.ffn.parameters(), lr = lr)
    projection_scheduler = torch.optim.lr_scheduler.StepLR(ensemble_optimizer, step_size=1, gamma=0.98)
    
    optimizers = {"ensemble":ensemble_optimizer, "cnn":cnn_optimizer, "ffn":ffn_optimizer}
    schedulers = {"ensemble":projection_scheduler}
    datasets = {"main":main_dataset, "ensemble":ensemble_dataset}
    dataloaders = {"main":main_dataloader, "ensemble":ensemble_dataloader}
    
    return model, optimizers, schedulers, datasets, dataloaders