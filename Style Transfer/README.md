# Style Transfer 

Neural style transfer is an optimization technique used to take two images—a content image and a style reference image (such as an artwork by a famous painter)—and blend them together so the output image looks like the content image, but “painted” in the style of the style reference image.

## Usage

1) Install the required libraries mentioned in requirement.txt.
2) Put the Image and the style image inside ```Input/``` folder and rename it as image.jpg and style.jpg.
3) Run ```python3 run.py``` and let the model train for about 100 iterations.
4) Check the output image inside the ```Output/``` folder.


## Model Architecure 

We use a Pre-Trained VGG19 model to optimize the image by taking output from various sections of our base model and optimizing the output.

## Output

!<img src="https://github.com/ShivamRajSharma/PyTorch/blob/master/Style%20Transfer/Input/henry.jpg" height="280" width ="200" />   +  !<img src="https://github.com/ShivamRajSharma/PyTorch/blob/master/Style%20Transfer/Input/starry_night.jpg" height="280" width ="200" />  =  !<img src="https://github.com/ShivamRajSharma/PyTorch/blob/master/Style%20Transfer/Output/generated_3800.png" height="280" width ="200"/>
