import warnings
warnings.filterwarnings('ignore')

import CONFIG
import CaptchaModel

import torch 
import torch.nn as nn
import numpy as np 
from PIL import Image
import pickle
import albumentations as alb 

def predict(image_path):

    idx_to_word = pickle.load(open("input/idx_to_word.pickle", "rb"))

    model = CaptchaModel.CaptchaModel(
        input_channels=CONFIG.input_channels,
        out_channels=CONFIG.out_channels, 
        kernel_size=CONFIG.kernel_size, 
        conv_dropout=CONFIG.conv_dropout,
        max_pool_size=CONFIG.max_pool_size,
        num_conv_layers=CONFIG.num_conv_layers,
        input_dims=CONFIG.input_dims, 
        hidden_dims=CONFIG.hidden_dims,
        num_layers=CONFIG.num_layers,
        rnn_dropout=CONFIG.rnn_dropout,
        num_classes=len(idx_to_word) + 1
    )

    model.load_state_dict(torch.load(CONFIG.MODEL_PATH))

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    transforms = alb.Compose([
        alb.Normalize(mean, std, always_apply=True),
        alb.Resize(CONFIG.image_height, CONFIG.image_width, always_apply=True)
    ])
    image = np.array(Image.open(image_path).convert("RGB"))
    image = transforms(image=image)['image']
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2)

    model.eval()
    with torch.no_grad():
        output = model(image)
        
    output = torch.argmax(output, dim=-1).squeeze(0)
    output = output.numpy()

    prediction = []
    for num, label in enumerate(output):
        if label == len(idx_to_word) or output[num]==output[num-1]:
            continue
        else:
            prediction.append(idx_to_word[label])
    
    prediction = ''.join(prediction)
    target = image_path.split('/')[-1].split('.')[0]

    print(f'TRUE -> {target} | PREDICTED -> {prediction}')


if __name__ == "__main__":
    predict('input/captcha_images_v2/4cn7b.png')
