import CONFIG
import UNet
import engine
import DataLoader

import numpy as np
import sys
import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import albumentations as alb

def run():
    path = CONFIG.INPUT_PATH 
    x_ray_image_names = os.listdir(path + '/CXR_png/')
    image_names = []
    for name in x_ray_image_names:
        image_names.append(name.split('.')[0])
    
    dataset_image_names = []
    mask_image_names = os.listdir(path + '/masks/')
    for name in mask_image_names:
        name = name.split('.png')[0].split('_mask')[0]
        if name in image_names:
            dataset_image_names.append(name)


    image_transforms = alb.Compose([
        alb.Normalize(CONFIG.mean, CONFIG.std, always_apply=True),
        alb.Resize(572, 572, always_apply=True),
        alb.pytorch.ToTensor()
    ])

    mask_transforms = alb.Compose([
        alb.Resize(388, 388, always_apply=True),
        alb.pytorch.ToTensor()
    ])

    train_images_name, val_images_name = train_test_split(dataset_image_names)

    train_data = DataLoader.DataLoader(
        train_images_name,
        image_transforms,
        mask_transforms
    )

    val_data = DataLoader.DataLoader(
        val_images_name,
        image_transforms,
        mask_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        num_workers=4,
        batch_size=CONFIG.Batch_size,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        num_workers=4,
        batch_size=CONFIG.Batch_size,
        pin_memory=True
    )

    # if torch.cuda.is_available():
    #     accelarator = 'cuda'
    #     torch.backends.cudnn.benchmark = True
    # else:
    #     accelarator = 'cpu'
    accelarator = 'cpu'
    
    device = torch.device(accelarator)

    model = UNet.UNet(input_channels=3)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        threshold=CONFIG.scheduler_thresh,
        mode='min'
    )

    best_loss = 1e-4
    
    print('------ [INFO] STARTING TRAINING ------')
    for epoch in range(CONFIG.Epochs):
        train_loss = engine.train_fn(model, train_loader, optimizer, device)
        val_loss = engine.eval_fn(model, val_loader, device)
        print(f'EPOCH -> {epoch}/{CONFIG.Epochs} | TRAIN LOSS = {train_loss} | VAL LOSS = {val_loss}')
        scheduler.step(val_loss)
        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            torch.save(CONFIG.MODEL_PATH)

if __name__ == "__main__":
    run()