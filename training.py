from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import euclidean_distances

from landmarks_utils import *
from cropping import *
from config import *
from face_landmarking_model import *
from face_dataset import *

# class SparsityLoss(nn.Module):
#     def __init__(self, max_non_zero_connections, threshold=0.001):
#         super(SparsityLoss, self).__init__()
#         self.max_non_zero_connections = max_non_zero_connections
#         self.threshold = threshold

#     def forward(self, weight_matrix):
#         # Calculate the L0 norm approximation using soft thresholding
#         soft_threshold = torch.abs(weight_matrix) - self.threshold
#         l0_norm = torch.sum(torch.relu(soft_threshold), dim=0)
        
#         # Calculate the excess connections beyond the desired limit
#         excess_connections = l0_norm - self.max_non_zero_connections
        
#         # Define a penalty term to encourage sparsity
#         penalty = torch.sum(torch.relu(excess_connections))
#         return penalty
    
    
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
                     num_parent_landmarks = 5, 
                     projectors=5, 
                     rotate=True, 
                     lr_projection=0.01,
                     lr_cnn=0.01,
                     lr_ffn=0.01):
    
    projection_mask = create_projection_mask(num_parent_landamrks=num_parent_landmarks)    
    model = FaceLandmarking(projection_mask, projectors=projectors).to(DEVICE)
    main_dataset = FaceDataset(model, subgroups = best_groups[:3], rotate=rotate)
    main_dataloader = DataLoader(main_dataset, batch_size=100, shuffle=True)

    ensemble_dataset = []
    ensemble_dataloader = []

    #print('creating datasets')
    for group in best_groups[:projectors]:
        simple_dataset = FaceDataset(model, subgroups=[group], rotate=rotate)
        ensemble_dataset.append(simple_dataset)
        ensemble_dataloader.append(DataLoader(simple_dataset, batch_size=50, sampler=EnsembleSampler(simple_dataset)))
    
    ensemble_optimizer = torch.optim.Adam(model.ensemble.parameters(), lr = lr_projection)
    cnn_optimizer = torch.optim.Adam(model.cnn_focusing.parameters(), lr = lr_cnn)
    ffn_optimizer = torch.optim.Adam(model.ffn.parameters(), lr = lr_ffn)
    projection_scheduler = torch.optim.lr_scheduler.StepLR(ensemble_optimizer, step_size=1, gamma=0.98)
    
    optimizers = {"ensemble": ensemble_optimizer, "cnn":cnn_optimizer, "ffn":ffn_optimizer}
    schedulers = {"ensemble": projection_scheduler}
    datasets = {"main": main_dataset, "ensemble": ensemble_dataset}
    dataloaders = {"main": main_dataloader, "ensemble": ensemble_dataloader}
    
    return model, optimizers, schedulers, datasets, dataloaders


def train (model, 
           optimizers, 
           schedulers, 
           datasets, 
           dataloaders, 
           pretraining = True,
           pretrain_epochs = 150,
           cnn_epochs = 0,
           ffn_epochs = 0,
           cnn_ffn_epochs = 0,
           all_train_epochs = 0):
     
    actual_optimizers = [optimizers["ensemble"]]
    TRAIN_PHASE = 0
    PRETRAINING = pretraining

    pretrain_cache = []
    main_cache = []
    # time_cache = {}
    # time_cache["dataset"] = []
    # time_cache["model"] = []
    # time_cache["backward"] = []

    #print('start training')
    for epoch in range(pretrain_epochs + cnn_epochs + ffn_epochs + cnn_ffn_epochs + all_train_epochs):

        if epoch == pretrain_epochs:
            PRETRAINING = False
            actual_optimizers = [optimizers["cnn"]]
            TRAIN_PHASE += 1
            print('\n Freezing raw_projection, training CNN_focusing.')

        if epoch == pretrain_epochs + cnn_epochs:
            actual_optimizers = [optimizers["ffn"]]
            TRAIN_PHASE += 1
            print('\n Freezing CNN_focusing, training top FFN.')

        if epoch == pretrain_epochs + cnn_epochs + ffn_epochs:
            actual_optimizers = [optimizers["cnn"], optimizers["ffn"]]
            print('\n Training CNN and FFN.')
        
        if epoch == pretrain_epochs + cnn_epochs + ffn_epochs + cnn_ffn_epochs:
            actual_optimizers = [optimizers["ensemble"], optimizers["cnn"], optimizers["ffn"]]
            print('\n Training all parameters.')

        datasets["main"].pretraining = PRETRAINING
        model.train_phase = TRAIN_PHASE
     
        if PRETRAINING:
            ensemble_inputs = []
            ensemble_targets = []
            for data_loader in dataloaders["ensemble"]:
                batch = next(iter(data_loader))
                inputs, targets, _ = batch
                ensemble_inputs.append(inputs)
                ensemble_targets.append(targets)
            
            for optimizer in actual_optimizers:
                optimizer.zero_grad()
            
            _, final_loss, raw_loss = model(
                x = ensemble_inputs,
                targets = ensemble_targets,
                multicrop = None,
            )
                     
            pretrain_cache.append(final_loss.item())
            final_loss.backward()

            for optimizer in actual_optimizers:
                optimizer.step()
            
            schedulers["ensemble"].step()
            
            if epoch > pretrain_epochs - 30:
                model.eval()
                batch = next(iter(dataloaders["main"]))
                inputs, targets, multicrop = batch
                
                _, total_loss, raw_loss = model(
                    x = inputs,
                    targets = targets,
                    multicrop = multicrop,
                )
                
                main_cache.append(total_loss.item())
                model.train()
                print(f"Pre-training epoch: {epoch}, loss: {total_loss.item()}.", end="\r")
            
            
        else:
            for iteration, batch in enumerate(dataloaders["main"]):
                for optimizer in actual_optimizers:
                    optimizer.zero_grad()
                
                # start = time()
                inputs, targets, multicrop = batch  
                # time_cache["dataset"].append(time() - start)
                
                # start = time()
                _, final_loss, raw_loss = model(
                    inputs,
                    targets = targets,
                    multicrop = multicrop,
                )
                # time_cache["model"].append(time() - start)
                
                
                main_cache.append(final_loss.item())
                # start = time()
                final_loss.backward()
                # time_cache["backward"].append(time() - start)
                
                print(f"Post-training epoch: {epoch}, final-loss: {final_loss.item()}, raw-loss: {raw_loss.item()}.", end="\r")

                for optimizer in actual_optimizers:
                    optimizer.step()
                
                # if iteration == 0 and epoch % 2 == 0:
                #     print(f'Epoch: {epoch}, iteration: {iteration}, final loss: {final_loss}, raw loss: {raw_loss}.')
        
    return pretrain_cache, main_cache#, time_cache