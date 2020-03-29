#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

#import matplotlib.pyplot as plt

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

def train_model(model,trainloader,testloader,criterion,optimizer,epochs,print_every,device):
    if (type(epochs)==type(None)):
        epochs=5
        print("Number of epochs specified as:",epochs)
    print("Device used for training is:",device)
    model.to(device)
    steps = 0
    running_loss = 0
    print("Print every used for testing:",print_every)
    print("Starting with Training of the model, with epochs:",epochs,"print after every:",print_every)
    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            steps+=1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
            if steps%print_every ==0:
                test_loss = 0
                accuracy=0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device),labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss+=batch_loss.item()
                    
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1)
                        equals=top_class==labels.view(*top_class.shape)
                        accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(testloader):.3f}.. "
                          f"Test accuracy: {accuracy/len(testloader):.3f}")
                    running_loss = 0
                    model.train()
    return model
    
def validate_model(model,testloader,criterion,device):
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device),labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss+=batch_loss.item()
                    
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1)
            equals=top_class==labels.view(*top_class.shape)
            accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
               
        print( f"Test accuracy: {100*accuracy/len(testloader):.3f}")
    model.train()
    return (accuracy/len(testloader))

def save_model(model,train_data,checkpoint_file,epochs):
    model.class_to_idx = train_data.class_to_idx
    print("Saving model .... \n\n", model, '\n')
    print("The state dict keys ... : \n\n", model.state_dict().keys())
    checkpoint = {'architecture': 'vgg16',
                  'epochs':epochs,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, checkpoint_file)
    return

def load_checkpoint(checkpoint_file):
    checkpoint = torch.load("checkpoint.pth")
    model=models.vgg16(pretrained=True)
    for param in model.parameters(): 
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def main():
    print("Image Classifier Training Module")
    parser=argparse.ArgumentParser()
    parser.add_argument("data_dir",help="Path to the directory which contains train/valid/test data")
    parser.add_argument("--save_dir",type=str,help="Path for saving checkpt file",default=os.getcwd())
    parser.add_argument("--arch",type=str,help="Specify the architecture for the model",default="vgg16")
    parser.add_argument("--learning_rate",type=float,help="Specify learning rate for the model training",default="0.001")
    parser.add_argument("--hidden_units",type=int,help="Specify the hidden units used for model training",default=4096)
    parser.add_argument("--epochs",type=int,help="Specify the epochs used for model training",default=5)
    parser.add_argument("--gpu",help="Turn on GPU mode",action='store_true',default=True)
    args=parser.parse_args()
    data_dir=args.data_dir
    save_dir=args.save_dir
    arch=args.arch
    learning_rate=args.learning_rate
    hidden_units=args.hidden_units
    epochs=args.epochs
    if args.gpu==True:
        device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    else:
        device = torch.device("cpu")
   
    print("Data Dir:",data_dir)
    print("Save Dir:",save_dir)
    print("Architecture:",arch)
    print("learning_rate:",learning_rate)
    print("hidden units:",hidden_units)
    print("Epochs:",epochs)
    print("device:",device)
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir,transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data,batch_size=50,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data,batch_size=50)
    testloader  = torch.utils.data.DataLoader(test_data,batch_size=50)
    images,labels = next(iter(trainloader))
    model=eval("models.{}(pretrained=True)".format(arch))
    model.name=arch 
    print("Standard classifier with model:",model)
    #Freeze parameters of the model
    for param in model.parameters():
        param.requires_grad = False
    #Create new classifier for the model
    classifier = nn.Sequential(OrderedDict([
                                          ('fc1', nn.Linear(25088,hidden_units,bias=True)),
                                          ('relu1', nn.ReLU()),
                                          ('dropout1', nn.Dropout(p=0.5)),
                                          ('fc2', nn.Linear(hidden_units,102,bias=True)),
                                          ('output', nn.LogSoftmax(dim=1))]))
    #Assign classifier to the model
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print("Updated classfier with the model:",model)
    #epochs=5
    print_every=30
    model=train_model(model,trainloader,validloader,criterion,optimizer,epochs,print_every,device)
    test_accuracy=0
    test_accuracy=validate_model(model,testloader,criterion,device)
    print('Accuracy achieved by the model on test images is: %d%%' % (100 * test_accuracy))
    #helper.imshow(images[0,:])
    checkpoint_file=save_dir+'/checkpoint.pth'
    save_model(model,train_data,checkpoint_file,epochs)
    
    
if __name__=="__main__":
    main()