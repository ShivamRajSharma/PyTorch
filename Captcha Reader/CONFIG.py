INPUT_PATH = 'input/captcha_images_v2/'
MODEL_PATH = 'model/model.bin'

Epochs = 100
Batch_Size = 32

val_size = 0.1

image_height = 50
image_width = 200
input_channels = 3
out_channels = 32
kernel_size = 3
conv_dropout = 0.0
num_conv_layers = 1
input_dims = out_channels*image_height
hidden_dims = 300
num_layers = 2
rnn_dropout = 0.0