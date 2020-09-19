num_workers = 4

blank_idx = 28

out_channel = 32
kernel_size = 4
padding = 1
num_inception_block = 1
squeeze_dims = 64
rnn_input_dims = squeeze_dims*out_channel
hidden_dims = 300
num_layers = 1
bidirectional = True
dropout = 0.3
num_classes = 29

n_filters = 128

dataset_path = '../input/'
MODEL_PATH = '../model/model.bin'

Epochs = 30
Batch_Size = 32