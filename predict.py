
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import argparse
import helper
import json
import PIL
import os
import seaborn as sns

def load_checkpoint(checkpoint_file):
    checkpoint = torch.load("checkpoint.pth")
    model=models.vgg16(pretrained=True)
    for param in model.parameters(): 
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    return ax

def predict(image_path,model,topk,device,cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to("cpu")
    # Set model to evaluate
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(topk)
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0] # This is not the correct way to do it but the correct way isnt working thanks to cpu/gpu issues so I don't care.
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    test_image = PIL.Image.open(image)

    # Get original dimensions
    orig_width, orig_height = test_image.size

    # Find shorter size and create settings to crop shortest side to 256
    if orig_width < orig_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]
        
    test_image.thumbnail(size=resize_size)

    # Find pixels to crop on to create 224x224 image
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))

    # Converrt to numpy - 244x244 image w/ 3 channels (RGB)
    np_image = np.array(test_image)/255 # Divided by 255 because imshow() expects integers (0:1)!!

    # Normalize each color channel
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
        
    # Set the color to the first channel
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

def main():
    print("Image Classifier Prediction Module")
    parser=argparse.ArgumentParser()
    parser.add_argument("path_to_image",help="Specify the full path for the image",default="flowers/test/100/image_07896.jpg")
    parser.add_argument("checkpoint_file",help="Specify the full path to checkpoint file",default="/home/workspace/ImageClassifier/checkpoint.pth")
    parser.add_argument("--category_name",type=str,help="Provide category name conversion file",default="cat_to_name.json")
    parser.add_argument("--topk",type=int,help="Specify K most likely classes",default=5)
    parser.add_argument("--gpu",help="Turn on GPU mode",action='store_true',default=True)
    args=parser.parse_args()
    imagefile = args.path_to_image
    checkpoint_file = args.checkpoint_file
    category_name=args.category_name
    topk=args.topk
    if args.gpu==True:
        device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    else:
        device = torch.device("cpu")
    print("Imagefile:",imagefile)
    print("Checkpointfile:",checkpoint_file)
    print("Category name:",category_name)
    print("topK:",topk)
    print("device:",device)
    model=load_checkpoint(checkpoint_file)
    with open(category_name, 'r') as f:
        cat_to_name = json.load(f)
 
    # Set up title
    flower_num = imagefile.split('/')[2]
    print("Flower Num:",flower_num)
    title_ = cat_to_name[flower_num]

    # Plot flower
    img = process_image(imagefile)

    # Make prediction
    probs, labs, flowers = predict(imagefile,model,topk,device,cat_to_name) 
    print("Flowers:",flowers)
    print("Probabilities:",probs)
    print("Labels:",labs)
    return

if __name__=="__main__":
    main()