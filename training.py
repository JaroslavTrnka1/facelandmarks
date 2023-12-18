from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from glob import glob

from landmarks_utils import *
from cropping import *
from config import *
from face_landmarking_model import *
from face_dataset import *
   
    
def create_projection_mask(dataset, num_parent_landmarks = 5, inputs = True):
    sample_n = 20
    cache = None
    for i in range(sample_n):
        idx = torch.randint(dataset.__len__(),(1,))
        parent_landmarks, children_landmarks  = dataset.get_landmarks(idx)    
        if not inputs:
            parent_landmarks = children_landmarks    
        similarity_matrix = euclidean_distances(children_landmarks.reshape(-1,2).numpy(), parent_landmarks.reshape(-1,2).numpy()).reshape(72,-1,1)
        if cache is None:
            cache=similarity_matrix
        else:
            cache=np.concatenate([cache, similarity_matrix], axis=2)
    avg_sim_matrix = np.mean(cache, axis = 2)
    mask = avg_sim_matrix < np.partition(avg_sim_matrix, num_parent_landmarks, axis=1)[:,num_parent_landmarks:num_parent_landmarks+1]
    return torch.from_numpy(mask).repeat_interleave(2, dim = 0).repeat_interleave(2, dim=1)

def create_template_mask(template_size, crop_size, gray=False, crop_as_template=False):
    if not gray:
        channels = 3
    else:
        channels = 1
    if crop_as_template:
        template_match_dim = (template_size - crop_size + 1)**2
    else:
        template_match_dim = (crop_size - template_size + 1)**2
    return torch.eye(72).repeat_interleave(2, dim = 0).repeat_interleave(template_match_dim*channels, dim=1).type(torch.float32)

def get_avg_template(dataset, template_size, template_sample = 50, gray = True):
    multicrops = []
    for i in range(template_sample):
        idx = torch.randint(dataset.__len__(),(1,))
        # targets, subimage, img_path = prep_dataset.__getitem__(idx)
        # img_blur = cv2.GaussianBlur(subimage, (3,3), 5) 
        # edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=150) 
        # image = edges[:,:,None]
        # multicrop = make_landmark_crops(targets.reshape(1,-1), image, template_size)
        x, y, multicrop = dataset.get_gray_multicrop(idx, template_size, gray=gray, from_targets=True)
        multicrops.append(normalize_multicrop(multicrop))

    return torch.mean(torch.stack(multicrops, dim=0), dim=0)


def get_best_groups():
    # Prepare best groups for ensemble projection
    groups = [os.path.basename(os.path.normpath(path_string)) for path_string in glob("./AI_Morphometrics/*/", recursive = False)]
    groups_results = pd.read_json('training_results_means.json', orient='index')
    groups_results.columns = ['result']
    list_df = pd.read_csv('./preprocessed_data/path_list.txt', names=['text'], header=None)
    counts = {}
    for group in groups:
        counts[group] = len(list_df[list_df['text'].str.contains(group)])
    compare = pd.concat([groups_results, pd.DataFrame(counts.items(), columns=['group', 'sample']).set_index('group')], axis = 1)
    #plt.scatter(x=compare['sample'], y=compare['result'])
    num_of_groups = 20
    best_groups = compare.dropna().sort_values(by=['sample'])[:num_of_groups].sort_values(by=['result']).index.to_list()
    return best_groups

def prepare_trainers(num_parent_landmarks = 5, 
                     projectors=5, 
                     rotate=True, 
                     lr_projection=0.01,
                     lr_cnn=0.01,
                     lr_ffn=0.01,
                     hidden_head_index=1,
                     ffn_bias=False,
                     template_size=50,
                     gray_scale = False,
                     template_method = None,
                     crop_as_template = False):
    
    print('Preparing masks and templates...')
    preparation_dataset = FaceDataset(rotate=rotate, gray=gray_scale)
    projection_mask = create_projection_mask(preparation_dataset, num_parent_landmarks=num_parent_landmarks)
    ffn_projection_mask = create_projection_mask(preparation_dataset, num_parent_landmarks=num_parent_landmarks, inputs=False)
    
    # TODO: crop_size as hyperparameter
    template_mask = create_template_mask(template_size, CROP_SIZE, gray=gray_scale, crop_as_template=crop_as_template)    
    avg_template = get_avg_template(preparation_dataset, template_size, gray=gray_scale)
    
    print('Preparing model, optimizers, datasets and dataloaders...')
    model = FaceLandmarking(
        projection_mask, 
        template_mask, 
        projectors=projectors, 
        hidden_head_index=hidden_head_index, 
        ffn_bias=ffn_bias,
        avg_template=avg_template,
        ffn_projection_mask=ffn_projection_mask,
        template_method=template_method,
        crop_as_template=crop_as_template
        ).to(DEVICE)
    
    del(preparation_dataset)
    
    main_dataset = FaceDataset(model, rotate=rotate, avg_template=avg_template, gray=gray_scale,crop_as_template=crop_as_template)
    main_dataloader = DataLoader(main_dataset, batch_size=100, shuffle=True)
    
    # TODO: rozdělit datasety
    test_dataset = FaceDataset(model, rotate=rotate, avg_template=avg_template, gray=gray_scale, template_method=template_method, crop_as_template=crop_as_template)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    ensemble_dataset = []
    ensemble_dataloader = []

    best_groups = get_best_groups()
    
    for group in range(projectors):
        simple_dataset = FaceDataset(model, subgroups=[best_groups[group]], rotate=rotate, avg_template=avg_template)
        ensemble_dataset.append(simple_dataset)
        ensemble_dataloader.append(DataLoader(simple_dataset, batch_size=200, sampler=EnsembleSampler(simple_dataset)))
    
    ensemble_optimizer = torch.optim.Adam(model.ensemble.parameters(), lr = lr_projection)
    cnn_optimizer = torch.optim.Adam(model.templ_match_projection.parameters(), lr = lr_cnn)
    #cnn_optimizer = torch.optim.Adam(model.cnn_focusing.parameters(), lr = lr_cnn)
    ffn_optimizer = torch.optim.Adam(model.ffn.parameters(), lr = lr_ffn)
    projection_scheduler = torch.optim.lr_scheduler.StepLR(ensemble_optimizer, step_size=1, gamma=0.98)
    
    optimizers = {"ensemble": ensemble_optimizer, "cnn":cnn_optimizer, "ffn":ffn_optimizer}
    schedulers = {"ensemble": projection_scheduler}
    datasets = {"main": main_dataset, "ensemble": ensemble_dataset, 'test': test_dataset}
    dataloaders = {"main": main_dataloader, "ensemble": ensemble_dataloader, 'test': test_dataloader}
    
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
    test_cache = []
    time_cache = {}
    time_cache["dataset"] = []
    time_cache["model"] = []
    time_cache["backward"] = []

    print('Start training!')
    for epoch in range(pretrain_epochs + cnn_epochs + ffn_epochs + cnn_ffn_epochs + all_train_epochs):

        if epoch == pretrain_epochs:
            PRETRAINING = False
            actual_optimizers = [optimizers["cnn"]]
            TRAIN_PHASE += 1
            print('\n Freezing raw_projection, training second module (CNN/template_match).')

        if epoch == pretrain_epochs + cnn_epochs:
            actual_optimizers = [optimizers["ffn"]]
            TRAIN_PHASE += 1
            print('\n Freezing second module, training top FFN.')

        if epoch == pretrain_epochs + cnn_epochs + ffn_epochs:
            actual_optimizers = [optimizers["cnn"], optimizers["ffn"]]
            print('\n Training CNN and FFN.')
        
        if epoch == pretrain_epochs + cnn_epochs + ffn_epochs + cnn_ffn_epochs:
            actual_optimizers = [optimizers["ensemble"], optimizers["cnn"], optimizers["ffn"]]
            print('\n Training all parameters.')

        datasets["main"].pretraining = PRETRAINING
        datasets["test"].pretraining = PRETRAINING
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
                targets = ensemble_targets
            )
                     
            pretrain_cache.append(final_loss.item())
            final_loss.backward()

            for optimizer in actual_optimizers:
                optimizer.step()
            
            schedulers["ensemble"].step()
            
            if epoch > pretrain_epochs - 30:
                model.eval()
                batch = next(iter(dataloaders["test"]))
                inputs, targets, _ = batch
                output = model.ensemble.predict(inputs)

                criterion = nn.MSELoss()
                loss = criterion(output, targets)
                
                test_cache.append(loss.item())
                model.train()
                print(f"Pre-training epoch: {epoch}, loss: {loss.item()}.", end="\r")
            
            
        else:
            start = time()
            for iteration, batch in enumerate(dataloaders["main"]):
                for optimizer in actual_optimizers:
                    optimizer.zero_grad()
                
                #start = time()
                #inputs, targets, multicrop = batch  
                inputs, targets, template_match = batch  
                time_cache["dataset"].append(time() - start)
                
                start = time()
                _, final_loss, raw_loss = model(
                    inputs,
                    targets = targets,
                    #multicrop = multicrop,
                    template_match = template_match
                )
                time_cache["model"].append(time() - start)
                
                
                main_cache.append(final_loss.item())
                start = time()
                final_loss.backward()
                time_cache["backward"].append(time() - start)
                
                print(f"Post-training epoch: {epoch}, final-loss: {final_loss.item()}, raw-loss: {raw_loss.item()}.", end="\r")

                for optimizer in actual_optimizers:
                    optimizer.step()
                    
                
                if iteration % 2 == 0:
                    model.eval()
                    batch = next(iter(dataloaders["test"]))
                    inputs, targets, template_match = batch
                    
                    _, total_loss, raw_loss = model(
                        x = inputs,
                        targets = targets,
                        #multicrop = multicrop,
                        template_match = template_match
                    )
                    
                    test_cache.append(total_loss.item())
                    model.train()
                    #print(f"Training epoch: {epoch}, test-loss: {total_loss.item()}.")
                    
                start = time()
                # if iteration == 0 and epoch % 2 == 0:
                #     print(f'Epoch: {epoch}, iteration: {iteration}, final loss: {final_loss}, raw loss: {raw_loss}.')
        
    return pretrain_cache, main_cache, test_cache, time_cache


