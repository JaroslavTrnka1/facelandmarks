import torch

# CROP_SIZE = 52
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BOTH_MODELS = False   # 478 + 68 !!!
STANDARD_IMAGE_WIDTH = 500

# Poměr template matching 40/20 fungoval dobře!
# při cropsize 30 a různých template_size celkově horší výsledky.
# při crop size 50 a template size 30 nebo 20 také dobré.
