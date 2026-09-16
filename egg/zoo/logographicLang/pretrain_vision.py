import time

import torch
import torch.nn.functional as F
from model_utilities.datasets.imagenet_sharded import make_wids_sampler
from model_utilities.datasets.imagenet_subsets_wids import ImageNet50WIDS
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

# save_dir = "data/cifar10"
# os.makedirs(save_dir, exist_ok=True)

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

NUM_CLASSES = 50

random_seed = 1
learning_rate = 0.001
num_epochs = 10
batch_size = 128

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])  


train_dataset = ImageNet50WIDS("/iridisfs/vlcgroup/vision_datasets/imagenet_webdataset/train", transform=transform_train)
test_dataset = ImageNet50WIDS("/iridisfs/vlcgroup/vision_datasets/imagenet_webdataset/val", transform=transform_test)
train_sampler = make_wids_sampler(train_dataset)
test_sampler = make_wids_sampler(test_dataset)


train_loader = DataLoader(dataset=train_dataset,
                          sampler=train_sampler,
                          batch_size=batch_size,
                          num_workers=8)

test_loader = DataLoader(dataset=test_dataset,
                         sampler=test_sampler,
                         batch_size=batch_size,
                         num_workers=8)

# for images, labels in train_loader:
#     print('Image batch dimensions:', images.shape)
#     print('Image label dimensions:', labels.shape)
#     print(labels)
#     break

model = models.resnet50(weights="IMAGENET1K_V2")

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_CLASSES)


# for param in model.parameters():
#     param.requires_grad = False



# model.classifier.requires_grad = True



# model.classifier[6] = nn.Sequential(
#                       nn.Linear(4096, 512),
#                       nn.ReLU(),
#                       nn.Dropout(0.4),
#                       nn.Linear(512, NUM_CLASSES))



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

torch.save(model.state_dict(), "/home/cd6g22/EGG/egg/zoo/logographicLang/pretrained-models/imageNet50_resnet50_train.pth")
torch.save(model.features, "/home/cd6g22/EGG/egg/zoo/logographicLang/pretrained-models/imageNet50_resnet50_features.pth")



