import os
import torch
from tqdm import tqdm
from glob import glob
import json

#from rembg import remove

from facelandmarks.config import *
from facelandmarks.cropping import *
from facelandmarks.landmarks_utils import *
from facelandmarks.preprocessing import *
from facelandmarks.face_landmarking_model import *
from facelandmarks.training import *
import itertools

# TODO: load model with args


results = {}
projectors = [3]
rotations = [True]
learning_rates = [0.001]
start_out_channels = [16]
num_parents = [10]
ffn_bias = [False]
kernel_sizes = [[5,3,3]]
pooling = [['max', 'avg']]  
num_cnn = [5]
# wing_loss_width = [1,2,10]
cnn_crop_size = [[70, 64, 56, 48, 40]]
crop_size = max([max(x) for x in cnn_crop_size])
hidden_per_lmark = [64]


logger_path = 'logger.csv'

with open(logger_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['params', 'epoch', 'iteration', 'loss', 'difference'])  # Write headers 

hyperparameter_combinations = list(itertools.product(learning_rates,
                                                     num_parents,
                                                     num_cnn,
                                                     start_out_channels,
                                                     ffn_bias,
                                                     kernel_sizes,
                                                     hidden_per_lmark,
                                                     cnn_crop_size))

i = 0
for lr, num_parent_landmarks, num_cnn, start_out_channels, ffn_bias, kernel_sizes, hidden_per_lmark, cnn_crop_size in tqdm(hyperparameter_combinations):
    combination = f"{lr}_{num_parent_landmarks}_{num_cnn}_{start_out_channels}_{ffn_bias}_{kernel_sizes}_{hidden_per_lmark}_{cnn_crop_size}"
    model, optimizers, schedulers, datasets, dataloaders = prepare_trainers(num_parent_landmarks=num_parent_landmarks, 
                                                                            projectors=3, 
                                                                            rotate=True, 
                                                                            lr_projection=0.01,
                                                                            lr_cnn=lr,
                                                                            lr_ffn=0.001,
                                                                            start_out_channels=start_out_channels,
                                                                            kernel_sizes=kernel_sizes,
                                                                            activations=False,
                                                                            pooling=pooling,
                                                                            crop_size=crop_size,
                                                                            batch_norm=False,
                                                                            num_cnn=num_cnn,
                                                                            cnn_crop_size = cnn_crop_size,
                                                                            hidden_per_lmark=hidden_per_lmark,
                                                                            mixture = False
                                                                            )
    
    
    cache = new_cache() 
    results[combination] = train(
                            model,
                            cache,
                            optimizers,
                            schedulers, 
                            datasets, 
                            dataloaders, 
                            pretrain_epochs=170, #170
                            cnn_epochs=20,
                            ffn_epochs=10,
                            all_train_epochs=0,
                            logger_path = logger_path,
                            combination = combination
                            )

    with open(f"result_{combination}_real_sequence.json", "w") as outfile: 
        json.dump(results[combination], outfile)
    
    print('Results were saved...')
                
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

torch.save(model.state_dict(), model_dir + '/saved_model.pt')
model_args = model.args
model_args['projection_mask'] = None
model_args['ffn_projection_mask'] = None

with open(model_dir + '/args.json', 'w') as file:
    json.dump(model_args, file, indent=4)

