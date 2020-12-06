INPUT_PATH = 'input/'
MODEL_PATH = 'model/model.bin'

LR = 1e-4
scheduler_thresh = 0.001
patience = 1
decay_factor=0.6
Epochs = 40
Batch_size = 2

mean = (0, 0, 0)
std = (1, 1, 1)

pred_threshold = 0.7
bce_loss_coeff = 0
dice_loss_coeff = 1
