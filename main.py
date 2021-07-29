import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import os
import numpy as np
from datetime import datetime
from torchsummary import summary
from tqdm import tqdm
import argparse

from model import VGGNet
from plotgraph import plotgraph


### description of model when saving plot graph and model file
parser = argparse.ArgumentParser(description="description of model")
parser.add_argument("--vgg_model_type", required=True, help="type of VGG model")
parser.add_argument("--description", default="")
args = parser.parse_args()



### parameters
model_name = "VGGNet"
lr = 1e-4
batch_size = 32
epochs = 1000
earlystop = 7  # for early stopping
# path = "D:/projects"
path = os.path.dirname(os.getcwd()) # os.getcwd = "D:/projects/GoogLeNet", os.path.dirname = "D:/projects" directory name of GoogLeNet
datapath = path + "/dataset"
description = args.vgg_model_type + args.description  # for more information in loss, acc graph and model file. ex) loss_"file_name".png, acc_"file_name".png in results folder & best_model_"file_name".h in model folder
resultpath = path + "/" + model_name + "/results"
modelpath = path + "/" + model_name + "/models/"
saved_model_name = modelpath + "best_model_" +  description + ".h" 
if not os.path.exists(resultpath):
      os.mkdir(resultpath)
if not os.path.exists(modelpath):
      os.mkdir(modelpath)
vgg_model_type = args.vgg_model_type

### gpu 
if torch.cuda.is_available():
      device = torch.device("cuda")
else:
      device = torch.device("cpu")
print("device:", device)



# ### dataset STL10
# data_transformer = transforms.Compose(
#   transforms.ToTensor()
# )
# train_set = datasets.STL10(datapath, split="train", download=True, transform=data_transformer)
# print(train_set.data.shape)

# # image Normalization
# # meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in train_set]
# # stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in train_set]
# # meanR = np.mean([m[0] for m in meanRGB])
# # meanG = np.mean([m[1] for m in meanRGB])
# # meanB = np.mean([m[2] for m in meanRGB])
# # stdR = np.mean([s[0] for s in stdRGB])
# # stdG = np.mean([s[1] for s in stdRGB])
# # stdB = np.mean([s[2] for s in stdRGB])
# # print("mean RGB:", meanR, meanG, meanB)
# # print("std RGB:", stdR, stdG, stdB)
# meanR, meanG, meanB = 0.4467106, 0.43980986, 0.40664646
# stdR, stdG, stdB = 0.22414584, 0.22148906,  0.22389975

# train_transformer = transforms.Compose([
#   transforms.Resize(256),
#   transforms.CenterCrop(224),
#   transforms.RandomHorizontalFlip(),
#   transforms.ToTensor(),
#   transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB]),
# ])
# train_set.transform = train_transformer

# # there is no validation set in STL10 dataset,
# # make validation set by splitting the train set.
# from sklearn.model_selection import StratifiedShuffleSplit
# # StratifiedShuffleSplit splits indices of train in same proportion of labels
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
# indices = list(range(len(train_set)))
# y_train0 = [y for _,y in train_set]
# for train_index, val_index in sss.split(indices, y_train0):
#       print('train :', len(train_index) , 'val :', len(val_index))
# # create two datasets from train_set
# from torch.utils.data import Subset
# val_set = Subset(train_set, val_index)
# train_set = Subset(train_set, train_index)
# # print(len(train_set), len(val_set))
# # count the number of images per calss in train_set and val_set
# # import collections
# # y_train = [y for _, y in train_set]
# # y_val = [y for _, y in val_set]
# # counter_train = collections.Counter(y_train)
# # counter_val = collections.Counter(y_val)
# # print(counter_train)
# # print(counter_val)

# ### dataloader for train_set, val_set
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
# # check dataloader
# # for x,y in train_loader:
# #     print(x.shape)
# #     print(y.shape)
# #     break
# # for a,b in val_loader:
# #     print(a.shape)
# #     print(b.shape)
# #     break


### dataset Cifar10
train_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.229, 0.224, 0.225)),
])
train_set = datasets.CIFAR10(root=datapath, train=True, download=True, transform=train_transform)
train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')






### model
model = VGGNet(model = vgg_model_type, in_channels=3, num_classes=10, init_weights=True).to(device)  ## CAUTION!!! test without auxiliary classifier!
print("model set to",next(model.parameters()).device)
# summary(model, input_size=(3, 224, 224))  # check model summary


### define loss function, optimizer, scheduler
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = lr)
# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)


### training function with validation per epoch
def train():
    model.train()
    loss_list, valloss_list, valacc_list = [], [], []  # lists for ploting graph
    best_acc = 0  # to update the best_acc to find whether early stop or not
    best_loss = float("inf")  # to update the best_acc to find whether early stop or not
    for epoch in range(epochs):
        avg_loss, val_loss, val_acc = 0, 0, 0  # initialize loss, acc

        for param_group in optimizer.param_groups:  # to see the learning rate per epoch
            current_lr =  param_group['lr']

        for x, y in tqdm(train_loader, leave=True):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()

            # forward propagation
            hypothesis = model(x)
            loss = criterion(hypothesis, y)

            # back propagation
            loss.backward()
            optimizer.step()

            avg_loss += loss/len(train_set)  # calculate average loss per epoch
        
        ### validation
        model.eval()
        print("========================validation========================")
        with torch.no_grad():
            for x, y in tqdm(val_loader, leave=True):
                x, y = x.to(device), y.to(device)

                prediction = model(x)

                # calculate validation Loss
                val_loss += criterion(prediction, y) / len(val_set)

                # calculate validation Accuracy
                val_acc += (prediction.max(1)[1] == y).sum().item() * 100 / len(val_set)

        print(datetime.now().time().replace(microsecond=0), "EPOCHS: [{}], current_lr: [{}], avg_loss: [{:.4f}], val_loss: [{:.4f}], val_acc: [{:.2f}%]".format(
                epoch+1, current_lr, avg_loss.item(), val_loss.item(), val_acc))

        # append list and plot graph
        loss_list.append(avg_loss.item())
        valloss_list.append(val_loss.item())
        valacc_list.append(val_acc)
        plotgraph(loss_list=loss_list, valloss_list=valloss_list, valacc_list=valacc_list, path = resultpath, description=description)

        # # Early Stop based on val_acc
        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     es = earlystop
        #     torch.save(model.state_dict(), saved_model_name)
        # else: 
        #     es -= 1
        # if es == 0: 
        #     # model.load_state_dict(torch.load(saved_model_name))
        #     print("Early Stopped and saved model")
        #     break

        # Early Stop based on val_loss
        if val_loss < best_loss:
            best_loss = val_loss
            es = earlystop
            torch.save(model.state_dict(), saved_model_name)
        else: 
            es -= 1
        if es == 0: 
            # model.load_state_dict(torch.load(model_save_name))
            print("Early Stopped and saved model")
            break

        # learning rate scheduler per epoch
        # lr_scheduler.step()
        lr_scheduler.step(val_loss)

    print("finished training")





### test function
def test():
    ### test
    ### test set for Cifar10
    test_transformer = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.229, 0.224, 0.225)),
                    transforms.Resize(224)
    ])
    test_set = datasets.CIFAR10(root=datapath, train=False,
                                        download=True, transform=test_transformer)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model.load_state_dict(torch.load(saved_model_name))
    model.eval()
    test_acc = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in tqdm(test_loader, leave=True):
            x, y = x.to(device), y.to(device)

            # prediction
            prediction = model(x)
            test_acc += (prediction.max(1)[1] == y).sum().item() * 100 / len(test_set)

    print("Acc: [{:.2f}%]".format(
        test_acc
    ))

# if __name__ == '__main__':
#     ### train
#     print("=====================start training=====================")
#     train()
#     ### test
#     test()

### train
print("=====================start training=====================")
train()
### test
test()