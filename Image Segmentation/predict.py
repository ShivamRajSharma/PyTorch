import CONFIG
import UNet

import torch 
import torch.nn as nn
import cv2
import numpy as np 
import albumentations as alb
from PIL import Image

def predict(image_path):
    model = UNet.UNet(input_channels=3)
    model.load_state_dict(torch.load(CONFIG.MODEL_PATH))
    model.eval()

    image = np.array(Image.open(image_path).convert('RGB'))
    mean = CONFIG.mean
    std = CONFIG.std

    transforms = alb.Compose([
        alb.Normalize(mean, std, always_apply=True),
        alb.Resize(512, 512, always_apply=True)
    ])

    image_ = transforms(image=image)['image']
    image_ = torch.tensor(image_).unsqueeze(0)
    image_ = image_.permute(0, 3, 1, 2)

    with torch.no_grad():
        prediction = model(image_)
    prediction = prediction.squeeze(0).squeeze(0)
    prediction = torch.sigmoid(prediction)
    prediction = (prediction>CONFIG.pred_threshold)*1
    prediction = prediction.detach().cpu().numpy()

    print('-- SAVING IMAGE --\n')

    image = cv2.resize(image, (prediction.shape[-1], prediction.shape[-1]))
    image[:, :, 1] = image[:, :, 1]*(1-prediction)

    cv2.imwrite('output/segmented.jpg', image)
    
    
if __name__ == "__main__":
    predict('input/CXR_png/CHNCXR_0001_0.png')
