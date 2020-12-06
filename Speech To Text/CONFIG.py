num_workers = 1

blank_idx = 28

transform_threshold = 0.1

LR = 5e-4

out_channel = 32
kernel_size = 3
padding = 1
num_inception_block = 2
squeeze_dims = 64
rnn_input_dims = squeeze_dims*out_channel
hidden_dims = 500
num_layers = 3
bidirectional = True
dropout = 0.3
num_classes = 29

scheduler_patience = 2
scheduler_threshold = 0.01

weight_decay = 0.0

n_filters = 128

dataset_path = 'input/'
MODEL_PATH = 'model/model.bin'

Epochs = 15
Batch_Size = 16
