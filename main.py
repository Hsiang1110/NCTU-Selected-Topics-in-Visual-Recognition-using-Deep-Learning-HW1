import torch
import torchvision.transforms as transforms
import torch.nn as nn
from model import net
import torch.optim as optim
import dataset
from train import train
import os
import csv
from PIL import Image

# Path
train_path = 'data/training_data/'
test_path = 'data/testing_data/'
train_label_path = 'data/training_labels.csv'

# Hyperparameter
batch_size = 24
lr = 0.01
total_epoch = 50

transform = transforms.Compose(
    [transforms.Resize((400, 400)),
     transforms.CenterCrop((384, 384)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transform_augmentation = transforms.Compose(
    [transforms.Resize((400, 400)),
     transforms.RandomResizedCrop((384, 384)),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# torch.utils.data.DataLoader
train_data = dataset.myImageFloder(root=train_path, label=train_label_path, transform=transform)
train_augmentation = dataset.myImageFloder(root=train_path, label=train_label_path, transform=transform_augmentation)
# ######################################Check training data###############################
# print(train_data.get_name(40))
# print(train_data.classes[train_data.__getitem__(40)[1]])
# print(train_data[40])
# ########################################################################################
train_size = int(0.7 * len(train_data))
valid_size = len(train_data) - train_size
train_data2, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])
concatDataset = torch.utils.data.ConcatDataset([train_data2, train_augmentation])
trainLoader = torch.utils.data.DataLoader(concatDataset, batch_size=batch_size, shuffle=True)
validLoader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)

# loss setting
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=0.0001)

# check GPU
if torch.cuda.is_available():
    print('GPU number: ', torch.cuda.device_count())  # 判断有多少 GPU
    print('GPU name:', torch.cuda.get_device_name(0))  # GPU名稱
    print('Current GPU: ', torch.cuda.current_device())  # GPU index
else:
    print('Use cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
print('device: ', device)

# train
print('Training')
train(net, total_epoch, trainLoader, validLoader, optimizer, lr, criterion, device)
# save model
PATH = 'model/net.pth'
torch.save(net.state_dict(), PATH)

# test
classes = train_data.classes
result = []  # predictions result
print('testing')
allFileList = os.listdir(test_path)
with torch.no_grad():
    for file in allFileList:
        img = Image.open(test_path + file).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        output = net(img)
        _, predicted = torch.max(output, 1)  # get the label with highest value
        result.append([file.split('.')[0], classes[predicted.item()]])

print('generate predictions.csv')
with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['id', 'label'])
    writer.writerows(result)
print('Finish')
