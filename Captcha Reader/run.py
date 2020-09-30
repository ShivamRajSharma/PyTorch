import CONFIG
import DataLoader 
import  CaptchaModel
import engine
import predict 

import numpy as np
import pickle
import torch 
import albumentations as alb

def run():
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transforms = alb.Compose([
        alb.Normalize(mean, std, always_apply=True),
        alb.Resize(50, 200, always_apply=True)
    ])
    
    dataset = DataLoader.DataLoader(transforms)
    pickle.dump(dataset.vocab.word_to_idx, open('input/word_to_idx.pickle', 'wb'))
    pickle.dump(dataset.vocab.idx_to_word, open('input/idx_to_word.pickle', 'wb'))

    dataset_size = int(len(dataset))
    indexex = list(range(dataset_size))
    train_index, val_index = indexex[int(CONFIG.val_size*dataset_size):], indexex[:int(CONFIG.val_size*dataset_size)]
    train_sampler = torch.utils.data.sampler.RandomSampler(train_index)
    val_sampler = torch.utils.data.sampler.RandomSampler(val_index)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CONFIG.Batch_Size,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CONFIG.Batch_Size,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler
    )

    # if torch.cuda.is_available():
    #     accelarator = 'cuda'
    #     torch.backends.cudnn.benchmark = True
    # else :
    #     accelarator = 'cpu'

    accelarator = 'cpu'
    
    device = torch.device(accelarator)

    num_classes = len(dataset.vocab.word_to_idx) + 1

    model = CaptchaModel.CaptchaModel(
        input_channels=CONFIG.input_channels,
        out_channels=CONFIG.out_channels, 
        kernel_size=CONFIG.kernel_size, 
        conv_dropout=CONFIG.conv_dropout,
        num_conv_layers=CONFIG.num_conv_layers,
        input_dims=CONFIG.input_dims, 
        hidden_dims=CONFIG.hidden_dims,
        num_layers=CONFIG.num_layers,
        rnn_dropout=CONFIG.rnn_dropout,
        num_classes=num_classes
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        threshold=0.01,
        mode='min'
    )
    blank = num_classes - 1


    best_loss = 1e4

    print('------ [INFO] STARTING TRAINING ------')
    for epoch in range(CONFIG.Epochs):
        train_loss = engine.train_fn(model, train_loader, optimizer, blank, device)
        val_loss = engine.eval_fn(model, val_loader, blank, device)
        scheduler.step(val_loss)
        print(f'EPOCH -> {epoch}/{CONFIG.Epochs} | TRAIN LOSS = {train_loss} | VAL LOSS = {val_loss}')
        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            predict.predict('input/captcha_images_v2/8y6b3.png')
            torch.save(best_model, CONFIG.MODEL_PATH)

if __name__ == "__main__":
    run()