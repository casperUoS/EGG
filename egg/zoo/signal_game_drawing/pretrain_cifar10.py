import os

import torch
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models

save_dir = "data/cifar10"
os.makedirs(save_dir, exist_ok=True)

# Print the full absolute path
print(f"Directory created/verified at: {os.path.abspath(save_dir)}")

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

NUM_CLASSES = 10

# Hyperparameters
random_seed = 1
learning_rate = 0.001
num_epochs = 10
batch_size = 128

custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

## Note that this particular normalization scheme is
## necessary since it was used for pre-training
## the network on ImageNet.
## These are the channel-means and standard deviations
## for z-score normalization.


train_dataset = datasets.CIFAR10(root='data',
                                 train=True,
                                 transform=custom_transform,
                                 download=True)

test_dataset = datasets.CIFAR10(root='data',
                                train=False,
                                transform=custom_transform)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          num_workers=8,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         num_workers=8,
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    print(labels)
    break

model = models.vgg19(pretrained=True)


for param in model.parameters():
    param.requires_grad = False

model.classifier.requires_grad = True



model.classifier[6] = nn.Sequential(
                      nn.Linear(4096, 512),
                      nn.ReLU(),
                      nn.Dropout(0.4),
                      nn.Linear(512, NUM_CLASSES))



model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


def compute_accuracy(model, data_loader):
    model.eval()
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        logits = model(features)
        _, predicted_labels = torch.max(logits, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


def compute_epoch_loss(model, data_loader):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            logits = model(features)
            e_loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += e_loss

        curr_loss = curr_loss / num_examples
        return curr_loss


losses = []
accuracies = []
start_time = time.time()
for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        ### FORWARD AND BACK PROP
        logits = model(features)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()

        loss.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        ### LOGGING
        if not batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                  % (epoch + 1, num_epochs, batch_idx,
                     len(train_loader), loss))

    model.eval()
    with torch.set_grad_enabled(False):  # save memory during inference
        loss_e = compute_epoch_loss(model, train_loader)
        acc_e = compute_accuracy(model, train_loader)
        losses.append(loss_e)
        accuracies.append(loss_e)
        print('Epoch: %03d/%03d | Train: %.3f%% | Loss: %.3f' % (
            epoch + 1, num_epochs,
            acc_e,
            loss_e))

    print("-" * 100)

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))



with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))



losses = torch.tensor(losses, device = 'cpu').tolist()
accuracies = torch.tensor(accuracies, device = 'cpu').tolist()

import matplotlib.pyplot as plt

plt.figure(figsize=[18,10])
plt.plot(losses, label="Train Loss")

plt.grid(True)
plt.legend()

plt.savefig("loss.jpg")

torch.save(model.state_dict(), "/home/casper/Documents/Github/EGG/egg/zoo/signal_game_drawing/data/cifar10/cifar10_vgg19_train.pth")
torch.save(model.features, "/home/casper/Documents/Github/EGG/egg/zoo/signal_game_drawing/data/cifar10/cifar10_vgg19_features.pth")



