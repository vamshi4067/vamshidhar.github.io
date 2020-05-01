---
title: "Malaria cell Detection"
date: 2019-12-21
tag: [Deep Learning,PyTorch]
---
# Detecting the cells effected from malaria so that the infected person can be informed at the initial stages of the infection before it turns endemic using Deep Learning


# importing the libraries

```python

import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
from os import listdir
import random
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as utils
```

Reading image files from the directory and creating the input and labels array
# converting to image size of 50*50
```python

shape = 50
images = []
labels = []
```
# infected cells
```python
path = '/Users/vamshi/Desktop/cell_images/'
infected_path = path+'Parasitized/'
for file in listdir(infected_path):
    if file.endswith('.png'):
        file_path = infected_path+file
        image = mpimg.imread(file_path)
        image = cv2.resize(image,(shape,shape))
        images.append(image)
        labels.append(1)
#uninfected cells
path = '/Users/vamshi/Desktop/cell_images/'
uninfected_path = path+'Uninfected/'
for file in listdir(uninfected_path):
    if file.endswith('.png'):
        file_path = uninfected_path+file
        image = mpimg.imread(file_path)
        image = cv2.resize(image,(shape,shape))
        images.append(image)
        labels.append(0)
        ```
# Shuffle cell images and their labels
```python
def reorder(old_list,order):
    new_list = []
    for i in order:
        new_list.append(old_list[i])
    return new_list

np.random.seed(seed=102)
index = np.arange(len(labels))
np.random.shuffle(index)
index = index.tolist()
labels = reorder(labels,index)
images = reorder(images,index)
```
# Visualizing first 10 images in dataset along with their labels
```python
def display_images(image_array,label):
    fig,axes = plt.subplots(2,5,figsize=(20,5))
    index = 0
    for i in range(2):
        for j in range(5):
            axes[i,j].imshow(image_array[index])
            axes[i,j].set_title(label[index],fontsize=20)
            index +=1
    plt.tight_layout()
    plt.show()

display_images(images[0:10],labels[0:10])
images = np.array(images)
labels = np.array(labels)
print(images.shape,labels.shape)
(27558, 50, 50, 3) (27558,)
```
# Convert to tensors and apply transforms
```python
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

img_dir='/Users/vamshi/Desktop/cell_images/'
train_data = datasets.ImageFolder(img_dir,transform=train_transforms)
```
# Splitting the data into train, test and validation set
```python

valid_size = 0.2
test_size = 0.3
total = len(train_data)
index = list(range(total))
np.random.shuffle(index)
valid_split = int(np.floor((valid_size) * total))
test_split = int(np.floor((valid_size+test_size) * total))
valid_idx, test_idx, train_idx = index[:valid_split], index[valid_split:test_split], index[test_split:]
print(len(valid_idx), len(test_idx), len(train_idx))
5511 8268 13779
```
# Loading into dataloaders
```python
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
    sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=32,
    sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(train_data, batch_size=20,
    sampler=test_sampler)
    ```
# Creating a pre-trained model Resnet18
```python
pretr_model = models.resnet18(pretrained=True)
pretr_model
pretr_model.fc
Linear(in_features=512, out_features=1000, bias=True)
```
# Changing the last layer so that it can be used for binary classification i.e infected or uninfected
```python
pretr_model.fc = nn.Linear(512,2)
```
# Turning on learning for parameters in last layer only
```python
for params in pretr_model.parameters():
    params.requires_grad = False

for params in pretr_model.fc.parameters():
    params.requires_grad = True
use_gpu = torch.cuda.is_available()
params_to_train = pretr_model.fc.parameters()
```
# use GPU if you have it
```python
if use_gpu:
    pretr_model = pretr_model.cuda()
```
# loss
```python
criterion = nn.CrossEntropyLoss()
```
# Create optimizer on the selected parameters
```python
optimizer_ft = optim.SGD(params_to_train, lr=0.01, momentum=0.9)
```
# Defining Training Function
```python

def train(n_epochs, model, optimizer, criterion, use_cuda,save_path):
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # train the model #
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # initialize weights to zero
            optimizer.zero_grad()

            output = model(data)

            # calculate loss
            loss = criterion(output, target)

            # back prop
            loss.backward()

            # grad
            optimizer.step()

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            if batch_idx % 100 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                  (epoch, batch_idx + 1, train_loss))

        # validate the model #
        model.eval()
        for batch_idx, (data, target) in enumerate(valid_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))


        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))

        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss

    # return trained model
    return model
    ```
# Training the model
```python
train(2,pretr_model,optimizer_ft,criterion,use_gpu,'trained_model')
Epoch 1, Batch 1 loss: 0.721161
Epoch 1, Batch 101 loss: 0.493954
Epoch 1, Batch 201 loss: 0.471339
Epoch: 1 	Training Loss: 0.466696 	Validation Loss: 0.401994
Validation loss decreased (inf --> 0.401994).  Saving model ...
Epoch 2, Batch 1 loss: 0.381501
Epoch 2, Batch 101 loss: 0.414743
Epoch 2, Batch 201 loss: 0.466714
Epoch: 2 	Training Loss: 0.466708 	Validation Loss: 0.603988
```
# Testing the data
```python
pretr_model.load_state_dict(torch.load('trained_model'))
<All keys matched successfully>
In [18]:
def test(model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    for batch_idx, (data, target) in enumerate(test_loader):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
test(pretr_model, criterion, use_gpu)
Test Loss: 0.400527


Test Accuracy: 83% (6887/8268)
```
# Visualize 10 predictions
```python

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)

def visualize(model,num_images=10):
    was_training = model.training
    model.eval()
    images = 0


    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
        # move to GPU
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            for j in range(data.size()[0]):
                images += 1
                fig = plt.figure(figsize=(10,10))
                ax = plt.subplot(num_images//2,2,images)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(pred[j].numpy()))
                imshow(data.cpu().data[j])

                if images == num_images:
                    model.train(mode=was_training)
                    return

visualize(pretr_model)
