import os
import albumentations as alb
from albumentations.pytorch import ToTensor


import torch 
import CONFIG
import torch.nn as nn 
import numpy as np
from PIL import Image

class  DataLoader(torch.utils.data.Dataset):
    def __init__(
        self, 
        image_name,
        image_transforms, 
        mask_transforms
    ):
        self.image_name = image_name
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
    
    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, idx):
        if self.image_name[idx][0] == 'M':
            extention = '.png'
        else:
            extention = '_mask.png'
        image = np.array(Image.open(os.path.join(CONFIG.INPUT_PATH, 'CXR_png', self.image_name[idx]+ '.png')).convert('RGB'))
        mask = np.array(Image.open(os.path.join(CONFIG.INPUT_PATH, 'masks', self.image_name[idx] + extention)))
        image = self.image_transforms(image=image)['image']
        mask = self.mask_transforms(image=mask)['image']
        mask = mask.unsqueeze(0)
        return {
            'original_image' : image,
            'mask' : mask
        }

if __name__ == "__main__":
    path = CONFIG.INPUT_PATH 
    x_ray_image_names = os.listdir(path + '/CXR_png/')
    images_name = []
    for name in x_ray_image_names:
        images_name.append(name.split('.')[0])
    
    image_transforms = alb.Compose([
        alb.Normalize(CONFIG.mean, CONFIG.std, always_apply=True),
        alb.Resize(572, 572, always_apply=True),
        alb.pytorch.ToTensor()
    ])
    segmented_transforms = alb.Compose([
        alb.Resize(388, 388, always_apply=True),
        ToTensor()
    ])
    dataloder = DataLoader(images_name, image_transforms, segmented_transforms)
    for data in dataloder:
        a = data['original_image']
        b = data['mask']
        print(a.shape, b.shape)
        break
