import time

import torch
import torch.nn.functional as F
from model_utilities.datasets.imagenet_sharded import make_wids_sampler
from model_utilities.datasets.imagenet_subsets_wids import ImageNet50WIDS
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import v2

# save_dir = "data/cifar10"
# os.makedirs(save_dir, exist_ok=True)

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

NUM_CLASSES = 50
REPO_DIR = "/home/cd6g22/EGG/dinov3"
vision_path ="/home/cd6g22/EGG/egg/zoo/logographicLang/pretrained-models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

random_seed = 1
learning_rate = 0.0005
num_epochs = 20
batch_size = 128

# transform_train = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#     transforms.RandomRotation(15),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225]),
# ])

# transform_test = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225]),
# ])  

transform_train = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    v2.RandomRotation(degrees=(-15.0, 15.0)),
    v2.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),),
])


transform_test = v2.Compose([
    v2.ToImage(),
    v2.Resize(256, antialias=True),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),)
])


train_dataset = ImageNet50WIDS("/iridisfs/vlcgroup/vision_datasets/imagenet_webdataset/train", transform=transform_train)
test_dataset = ImageNet50WIDS("/iridisfs/vlcgroup/vision_datasets/imagenet_webdataset/val", transform=transform_test)
train_sampler = make_wids_sampler(train_dataset)
test_sampler = make_wids_sampler(test_dataset)


train_loader = DataLoader(dataset=train_dataset,
                          sampler=train_sampler,
                          batch_size=batch_size,
                          num_workers=8,
                          persistent_workers=True,
                          pin_memory=True)

test_loader = DataLoader(dataset=test_dataset,
                         sampler=test_sampler,
                         batch_size=batch_size,
                         num_workers=8,
                         persistent_workers=True,
                         pin_memory=True)

# for images, labels in train_loader:
#     print('Image batch dimensions:', images.shape)
#     print('Image label dimensions:', labels.shape)
#     print(labels)
#     break

# model = models.resnet50(weights="IMAGENET1K_V2")




class DinoClassifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.dino = model
        self.lin1 = nn.Sequential(nn.Linear(768, 256), nn.ReLU())
        self.classifier =  nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
       x = self.dino(x)
       x = self.lin1(x)
       x = self.classifier(x)
       return x

# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, NUM_CLASSES)


# for param in model.parameters():
#     param.requires_grad = False



# model.classifier.requires_grad = True



# model.classifier[6] = nn.Sequential(
#                       nn.Linear(4096, 512),
#                       nn.ReLU(),
#                       nn.Dropout(0.4),
#                       nn.Linear(512, NUM_CLASSES))

dino = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=vision_path)

model = DinoClassifier(dino)

model = model.to(DEVICE)

for p in model.dino.parameters():
    p.requires_grad = False
    
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


def compute_accuracy(model, data_loader):
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.inference_mode():
        for features, targets in data_loader:
            features = features.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            logits = model(features)
            predicted_labels = logits.argmax(dim=1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum().item()
    return 100 * correct_pred / num_examples


losses = []
accuracies = []
start_time = time.time()
for epoch in range(num_epochs):

    model.train()
    epoch_loss_sum = 0.0
    epoch_correct = 0
    epoch_examples = 0
    for batch_idx, (features, targets) in enumerate(train_loader):

        features = features.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        ### FORWARD AND BACK PROP
        logits = model(features)
        loss = F.cross_entropy(logits, targets)
        epoch_loss_sum += loss.detach().item() * targets.size(0)
        epoch_correct += (logits.detach().argmax(dim=1) == targets).sum().item()
        epoch_examples += targets.size(0)
        optimizer.zero_grad()

        loss.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        ### LOGGING
        if not batch_idx % 50:
            print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} | '
                f'Batch {batch_idx:04d}/{len(train_loader):04d} | '
                f'Loss: {loss:.4f}', flush=True)

    loss_e = epoch_loss_sum / epoch_examples
    acc_e = 100 * epoch_correct / epoch_examples
    losses.append(loss_e)
    accuracies.append(acc_e)
    print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} | '
        f'Train: {acc_e:.3f}% | Loss: {loss_e:.3f}', flush=True)

    print("-" * 100)

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))



with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: {:.2f}%'.format(compute_accuracy(model, test_loader)))



losses = torch.tensor(losses, device = 'cpu').tolist()
accuracies = torch.tensor(accuracies, device = 'cpu').tolist()

import matplotlib.pyplot as plt

plt.figure(figsize=[18,10])
plt.plot(losses, label="Train Loss")

plt.grid(True)
plt.legend()

plt.savefig("loss.jpg")

torch.save(model.state_dict(), "/home/cd6g22/EGG/egg/zoo/logographicLang/pretrained-models/dinov3_16b_50c_20epochs.pth")
# torch.save(model.features, "/home/cd6g22/EGG/egg/zoo/logographicLang/pretrained-models/imageNet50_resnet50_features.pth")



