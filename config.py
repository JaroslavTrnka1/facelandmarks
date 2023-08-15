import torch

CROP_SIZE = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BOTH_MODELS = False