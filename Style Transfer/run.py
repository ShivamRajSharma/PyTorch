import torch 
from tqdm import tqdm
import torch.nn as nn
import torchvision
import numpy as np 
from PIL import Image
import albumentations as alb
from torchvision.utils import save_image


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.out_features_number = [0, 5, 10, 19, 28]
        self.base_model = torchvision.models.vgg19(pretrained=True).features[:29]
    
    def forward(self, x):
        features = []
        for num, layer in enumerate(self.base_model):
            x = layer(x)
            if num in self.out_features_number:
                features.append(x)
        return features


def augentation(data):
    imagenet_mean = (0.5071, 0.4867, 0.4408)
    image_net_std = (0.2675, 0.2565, 0.2761)
    transforms = alb.Compose(
        [
        alb.Normalize((0, 0, 0), (1, 1, 1), max_pixel_value=255, always_apply=True),
        alb.Resize(420, 420, always_apply=True),
        ],
        additional_targets={"image2" : "image"}
    )
    image = transforms(**data)

    return image


def image_loader(real_image_path, style_image_path):
    image = np.array(Image.open(real_image_path).convert("RGB"))
    image2 = np.array(Image.open(style_image_path).convert("RGB"))
    aug_input = {
        'image': image,
        'image2' : image2
    }
    image = augmentation(aug_input)
    return image


image_name = 'henry.jpg'
style_name = 'starry_night.jpg'

real_image_path =  f'Input/{image_name}'
style_image_path = f'Input/{style_name}'

steps = 4000
alpha = 0.98
beta = 0.02

if torch.cuda.is_available():
    compute = 'cuda'
    torch.backends.cudnn.benchmark=True
else:
    compute = 'cpu'

device = torch.device(compute)

image = image_loader(real_image_path, style_image_path)
real_img = torch.tensor(image['image'].transpose(2, 0, 1)).unsqueeze(0).to(device)
style_img = torch.tensor(image['image2'].transpose(2, 0, 1)).unsqueeze(0).to(device)

generated_img = real_img.clone().requires_grad_(True)

optimizer = torch.optim.Adam([generated_img], lr=0.001)

model = model().to(device).eval()

for step in tqdm(range(steps)):
    generated_img.data.clamp_(0, 1)
    style_features = model(style_img)
    generated_features = model(generated_img)
    original_features = model(real_img)

    style_loss, orig_loss = 0, 0

    for generated_feature, original_feature, style_feature in zip(
        generated_features, original_features, style_features
    ):
        batch_size, channel, height, width = generated_feature.shape
        
        orig_loss += torch.mean((generated_feature- original_feature)**2)

        gen_gram_matrix = torch.mm(generated_feature.view(channel, height*width), generated_feature.view(channel, height*width).t())
        style_gram_matrix = torch.mm(style_feature.view(channel, height*width), style_feature.view(channel, height*width).t())
        
        style_loss += torch.mean((gen_gram_matrix - style_gram_matrix)**2)
    
    total_loss = alpha*orig_loss + beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step%100 == 0:
        save_image(generated_img, f'Output/generated_{step}.png')
