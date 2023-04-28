from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from train import Net

import torch
from PIL import Image
import torchvision.transforms as transforms

def inference(path):

    # open method used to open different extension image file
    im = Image.open(path)
    #im = Image.open(r"C:\Users\amyss\OneDrive\Documents\nyu-2\cloud_ml\hw5\src\five.png") 
    
    # Define a transform to convert PIL 
    # image to a Torch tensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.PILToTensor(),
        transforms.Resize((28,28))
    ])
    
    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(im).unsqueeze(0)
    
    # print the converted Torch tensor
    #print(img_tensor)
    #print(img_tensor.size())

    if torch.cuda.is_available():
        device = device = torch.device("cuda")
    else :
        device = torch.device("cpu")

    model = Net().to(device)
    model.load_state_dict(torch.load(r"C:\Users\amyss\OneDrive\Documents\nyu-2\cloud_ml\hw5\src\mnist_cnn.pt"))
    model.eval()

    data = img_tensor.to(device)
    output = model(data.float())
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    print(pred)
    return pred
    

if __name__ == '__main__':
    inference()