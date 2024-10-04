from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from glob import glob
import csv
from itertools import chain
from facelandmarks.landmarks_utils import *
from facelandmarks.cropping import *
from facelandmarks.config import *
from facelandmarks.face_landmarking_model import *
from facelandmarks.face_dataset import *

if 'cuda' in str(DEVICE):
    from torch.cuda.amp import GradScaler, autocast
   
def new_cache():
    cache = {}
    cache['pretrain_cache'] = []
    cache['main_cache'] = []
    cache['improvements_cache'] = []
    cache['test_cache'] = []
    cache['lr'] = []
    cache['time_cache'] = {
        "dataset": [],
        "model": [],
        "backward": []
    }
    return cache
    
def create_projection_mask(dataset, num_parent_landmarks = 5, inputs = True):
    sample_n = 20
    cache = None
    for i in range(sample_n):
        idx = torch.randint(dataset.__len__(),(1,))
        parent_landmarks, children_landmarks  = dataset.get_landmarks(idx)    
        if not inputs:
            parent_landmarks = children_landmarks    
        similarity_matrix = euclidean_distances(children_landmarks.reshape(-1,2).cpu().numpy(), parent_landmarks.reshape(-1,2).cpu().numpy()).reshape(72,-1,1)
        if cache is None:
            cache=similarity_matrix
        else:
            cache=np.concatenate([cache, similarity_matrix], axis=2)
    avg_sim_matrix = np.mean(cache, axis = 2)
    mask = avg_sim_matrix < np.partition(avg_sim_matrix, num_parent_landmarks, axis=1)[:,num_parent_landmarks:num_parent_landmarks+1]
    return torch.from_numpy(mask).repeat_interleave(2, dim = 0).repeat_interleave(2, dim=1).to(torch.int8)

# def create_template_mask(template_size, crop_size, gray=False, crop_as_template=False):
#     if not gray:
#         channels = 3
#     else:
#         channels = 1
#     if crop_as_template:
#         template_match_dim = (template_size - crop_size + 1)**2
#     else:
#         template_match_dim = (crop_size - template_size + 1)**2
#     return torch.eye(72).repeat_interleave(2, dim = 0).repeat_interleave(template_match_dim*channels, dim=1).type(torch.float32)

# def get_avg_template(dataset, template_size, template_sample = 50, gray = True):
#     multicrops = []
#     for i in range(template_sample):
#         idx = torch.randint(dataset.__len__(),(1,))
#         # targets, subimage, img_path = prep_dataset.__getitem__(idx)
#         # img_blur = cv2.GaussianBlur(subimage, (3,3), 5) 
#         # edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=150) 
#         # image = edges[:,:,None]
#         # multicrop = make_landmark_crops(targets.reshape(1,-1), image, template_size)
#         x, y, multicrop = dataset.get_gray_multicrop(idx, template_size, gray=gray, from_targets=True)
#         multicrops.append(normalize_multicrop(multicrop))

#     return torch.mean(torch.stack(multicrops, dim=0), dim=0)


def get_best_groups():
    # Prepare best groups for ensemble projection
    groups = [os.path.basename(os.path.normpath(path_string)) for path_string in glob("./data/*/*", recursive = False)]
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
    return best_groups[1:]   # Due to colab training

def prepare_trainers(num_parent_landmarks = 5, 
                     projectors=5, 
                     rotate=True, 
                     lr_projection=0.01,
                     lr_cnn=0.01,
                     lr_ffn=0.01,
                     start_out_channels=8,
                     ffn_bias=False,
                     gray_scale = False,
                     kernel_sizes = [3,3,3],
                     activations = True,
                     pooling = True,
                     padding=0,
                     crop_size = 46,
                     batch_norm = False,
                     num_cnn = 5,
                     wing_loss_width=5,
                     cnn_crop_size = None,
                     hidden_per_lmark=64,
                     mixture=None,
                     top_head=None):
    
    seed = 42
    torch.manual_seed(seed)
    
    print('Preparing masks ...')    
    preparation_dataset = FaceDataset(rotate=rotate, gray=gray_scale)
    projection_mask = create_projection_mask(preparation_dataset, num_parent_landmarks=5)
    ffn_projection_mask = create_projection_mask(preparation_dataset, num_parent_landmarks=num_parent_landmarks, inputs=False)
    # But only raw projection for parent landmarks!
    ffn_projection_mask = ffn_projection_mask.repeat(1,num_cnn+1)
    
    print('Preparing model, optimizers, datasets and dataloaders...')
    best_groups = get_best_groups()
    
    if cnn_crop_size is None:
        cnn_crop_size = crop_size
    
    model = FaceLandmarking(
        projection_mask,  
        projectors=projectors, 
        start_out_channels=start_out_channels, 
        ffn_projection_mask=ffn_projection_mask,
        kernel_sizes=kernel_sizes,
        activations=activations,
        pooling=pooling,
        padding=padding,
        crop_size=cnn_crop_size,
        batch_norm = batch_norm,
        num_cnn=num_cnn,
        wing_loss_width=wing_loss_width,
        hidden_per_lmark=hidden_per_lmark,
        mixture = mixture,
        top_head = top_head
        ).to(torch.float32).to(DEVICE)  
    
    # scaler = GradScaler() 
    print('Model prepared...')
    del(preparation_dataset)
    
    base_dataset = FaceDataset(model, rotate=rotate, crop_size=crop_size)
    train_ratio = 0.95
    train_size = int(train_ratio * len(base_dataset))
    eval_size = len(base_dataset) - train_size
    main_dataset, test_dataset = random_split(base_dataset, [train_size, eval_size])
    
    # test_dataset = FaceDataset(model, rotate=rotate, crop_size=crop_size)#, subgroups=best_groups[:10])
    # main_dataset = FaceDataset(model, rotate=rotate, crop_size=crop_size)#, subgroups=best_groups[:10])
    
    main_dataloader = DataLoader(main_dataset, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    ensemble_dataset = []
    ensemble_dataloader = []
    for group in range(projectors):
        simple_dataset = FaceDataset(model, rotate=rotate, subgroups=[best_groups[group]])
        ensemble_dataset.append(simple_dataset)
        ensemble_dataloader.append(DataLoader(simple_dataset, batch_size=200, sampler=EnsembleSampler(simple_dataset)))
    print('Datsets prepared... preparing optimizers')
    
    ensemble_optimizer = torch.optim.Adam(model.ensemble.parameters(), lr = lr_projection)
    cnn_optimizer = torch.optim.Adam(model.focusing.parameters(),lr = lr_cnn)

    # cnn_optimizer2 = torch.optim.Adam(model.cnn_ensemble2.parameters(),lr = lr_cnn)
    if model.ffn is not None:
        ffn_optimizer = torch.optim.Adam(model.ffn.parameters(), lr = lr_ffn)
    else:
        ffn_optimizer = None
    projection_scheduler = torch.optim.lr_scheduler.StepLR(ensemble_optimizer, step_size=1, gamma=0.98)
    # cnn_scheduler = torch.optim.lr_scheduler.StepLR(cnn_optimizer, step_size=1, gamma=0.99)
    # cnn_scheduler = torch.optim.lr_scheduler.OneCycleLR(cnn_optimizer, max_lr=lr_cnn, pct_start= 0.1, steps_per_epoch=len(main_dataloader), epochs=20,anneal_strategy='linear')
    cnn_scheduler1 = torch.optim.lr_scheduler.LinearLR(cnn_optimizer, start_factor = 0.1, end_factor = 1, total_iters = 10 * len(main_dataloader) // 10)
    cnn_scheduler2 = torch.optim.lr_scheduler.LinearLR(cnn_optimizer, start_factor = 1, end_factor = 0.1, total_iters = 20*len(main_dataloader) - (20*len(main_dataloader)//10))
    cnn_scheduler = torch.optim.lr_scheduler.SequentialLR(cnn_optimizer, schedulers=[cnn_scheduler1, cnn_scheduler2], milestones=[20 * len(main_dataloader) // 10])
    if model.ffn is not None:
        ffn_scheduler1 = torch.optim.lr_scheduler.LinearLR(ffn_optimizer, start_factor = 0.1, end_factor = 1, total_iters = 10 * len(main_dataloader) // 10)
        ffn_scheduler2 = torch.optim.lr_scheduler.LinearLR(ffn_optimizer, start_factor = 1, end_factor = 0.1, total_iters = 10*len(main_dataloader) - (10*len(main_dataloader)//10))
        ffn_scheduler = torch.optim.lr_scheduler.SequentialLR(ffn_optimizer, schedulers=[ffn_scheduler1, ffn_scheduler2], milestones=[10 * len(main_dataloader) // 10])
    else:
        ffn_scheduler = None
        
    optimizers = {"ensemble": ensemble_optimizer, "cnn":cnn_optimizer,  "ffn":ffn_optimizer} #"cnn2":cnn_optimizer2,
    schedulers = {"ensemble": projection_scheduler, "cnn": cnn_scheduler, "ffn": ffn_scheduler}
    datasets = {"main": main_dataset, "ensemble": ensemble_dataset, 'test': test_dataset}
    dataloaders = {"main": main_dataloader, "ensemble": ensemble_dataloader, 'test': test_dataloader}
    
    return model, optimizers, schedulers, datasets, dataloaders


def train (model, 
           cache,
           optimizers, 
           schedulers, 
           datasets, 
           dataloaders, 
           pretraining = True,
           pretrain_epochs = 170,
           cnn_epochs = 0,
           ffn_epochs = 0,
           cnn_ffn_epochs = 0,
           all_train_epochs = 0,
           scaler = None,
           logger_path = None,
           combination = None):
    
     
    actual_optimizers = [optimizers["ensemble"]]
    TRAIN_PHASE = 0
    PRETRAINING = pretraining

    print('Start training!')
    for epoch in range(pretrain_epochs + cnn_epochs + ffn_epochs + cnn_ffn_epochs + all_train_epochs - 1):

        if (epoch == pretrain_epochs) and ((cnn_epochs + ffn_epochs + cnn_ffn_epochs + all_train_epochs) > 0):
            PRETRAINING = False
            actual_optimizers = [optimizers["cnn"]]
            scheduler = schedulers["cnn"]
            TRAIN_PHASE += 1
            print('\n Freezing raw_projection, training second module (CNN).')

        if (epoch == (pretrain_epochs + cnn_epochs)) and ((ffn_epochs + all_train_epochs) > 0):
            actual_optimizers = [optimizers["ffn"]]
            scheduler = schedulers["ffn"]
            TRAIN_PHASE += 1
            print('\n Freezing second module, training top FFN.')

        if (epoch == (pretrain_epochs + cnn_epochs + ffn_epochs)) and (all_train_epochs > 0):
            actual_optimizers = [optimizers["cnn"], optimizers["ffn"]]
            print('\n Training CNN and FFN.')
        
        if epoch == (pretrain_epochs + cnn_epochs + ffn_epochs + cnn_ffn_epochs):
            actual_optimizers = [optimizers["ensemble"], optimizers["cnn"], optimizers["ffn"]]
            print('\n Training all parameters.')

        datasets["main"].dataset.pretraining = PRETRAINING
        datasets["test"].dataset.pretraining = PRETRAINING
        # datasets["main"].pretraining = PRETRAINING
        # datasets["test"].pretraining = PRETRAINING
        model.train_phase = TRAIN_PHASE
     
        if PRETRAINING:
            ensemble_inputs = []
            ensemble_targets = []
            for data_loader in dataloaders["ensemble"]:
                batch = next(iter(data_loader))
                inputs, targets, _ = batch
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                ensemble_inputs.append(inputs)
                ensemble_targets.append(targets)
            
            for optimizer in actual_optimizers:
                optimizer.zero_grad()
            
            _, final_loss, raw_loss = model(
                x = ensemble_inputs,
                targets = ensemble_targets
            )
            
            # with autocast():
            #     _, final_loss, raw_loss = model(
            #         x=ensemble_inputs,
            #         targets=ensemble_targets
            #     )
                     
            cache['pretrain_cache'].append(final_loss.item())
            final_loss.backward()
            # scaler.scale(final_loss).backward()

            for optimizer in actual_optimizers:
                optimizer.step()
                # scaler.step(optimizer)
                
            # scaler.update()
            
            schedulers["ensemble"].step()
            
            if epoch > pretrain_epochs - 30:
                model.eval()
                batch = next(iter(dataloaders["test"]))
                inputs, targets, _ = batch
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                output = model.ensemble.predict(inputs)

                criterion = nn.MSELoss()
                loss = criterion(output, targets)
                
                cache['test_cache'].append(loss.item())
                model.train()
                print(f"Pre-training epoch: {epoch}, loss: {loss.item()}.", end="\r")
            
            
        else:
            start = time()
            for iteration, batch in enumerate(dataloaders["main"]):
                model.train()
                for optimizer in actual_optimizers:
                    optimizer.zero_grad()
                
                #start = time()
                #inputs, targets, multicrop = batch  
                inputs, targets, multicrop, landmarks, heatmaps, image_sizes = batch  
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                multicrop = multicrop.to(DEVICE)
                landmarks = landmarks.to(DEVICE)
                heatmaps = heatmaps.to(DEVICE)
                image_sizes = image_sizes.to(DEVICE)
                cache['time_cache']["dataset"].append(time() - start)
                
                start = time()
                # with autocast():
                _, final_loss, raw_loss = model(
                    inputs,
                    targets = targets,
                    multicrop = multicrop,
                    landmarks = landmarks,
                    heatmaps_true = heatmaps,
                    image_sizes = image_sizes
                )

                cache['time_cache']["model"].append(time() - start)
                
                cache['main_cache'].append(final_loss.item())   

                start = time()
                final_loss.backward()
                # scaler.scale(final_loss).backward()
                cache['time_cache']["backward"].append(time() - start)
                
                with open(logger_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([combination, epoch, iteration, final_loss.item(), raw_loss.item() - final_loss.item()])
                
                print(f"Post-training epoch: {epoch}, final-loss: {final_loss.item()}, raw-loss: {raw_loss.item()}.", end="\r")

                # for optimizer in actual_optimizers:
                #     scaler.step(optimizer)
                # scaler.update()
                
                for optimizer in actual_optimizers:
                    optimizer.step()
                
                cache['lr'].append(scheduler.get_last_lr()[0])
                if model.train_phase == 1:
                    scheduler.step()    
                
                if iteration % 5 == 0:
                    model.eval()
                    batch = next(iter(dataloaders["test"]))
                    inputs, targets, multicrop, landmarks, heatmaps, image_sizes = batch  
                    inputs = inputs.to(DEVICE)
                    targets = targets.to(DEVICE)
                    multicrop = multicrop.to(DEVICE)
                    landmarks = landmarks.to(DEVICE)
                    heatmaps = heatmaps.to(DEVICE)
                    image_sizes = image_sizes.to(DEVICE)
                    
                    # with autocast():
                    _, total_loss, raw_loss = model(
                        x = inputs,
                        targets = targets,
                        multicrop = multicrop,
                        landmarks = landmarks,
                        heatmaps_true = heatmaps,
                        image_sizes = image_sizes
                    )
                    
                    cache['test_cache'].append(total_loss.item())
                    cache['improvements_cache'].append(raw_loss.item() - total_loss.item())
                    model.train()
                    print('')
                    print(f"Training epoch: {epoch}, improvement: {raw_loss.item() - total_loss.item()}, final loss: {total_loss.item()}, raw loss: {raw_loss.item()}.")
                    
                start = time()
                # if iteration == 0 and epoch % 2 == 0:
                #     print(f'Epoch: {epoch}, iteration: {iteration}, final loss: {final_loss}, raw loss: {raw_loss}.')
    return cache

