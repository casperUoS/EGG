
import numpy
import numpy as np
import torch
from gensim.models import KeyedVectors
from numpy.linalg import norm
from scipy import stats
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models


class SketchDataset(Dataset):
    def __init__(self, messages, labels):
        """
        :param messages: Tensor of shape (num_samples, height, width, channels)
        """
        self.messages = messages
        self.labels = labels

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        message = self.messages[idx]
        label = self.labels[idx]
        return message, label


def semantic_correlaton( vision, val_dataloader):

    cate_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

    word2vec = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

    cate2vec = {}
    for name in cate_names:
        cate2vec[name] = numpy.asarray(word2vec[name], dtype=np.float32)

    feature_extractor = vision.features

    cate_features = {}

    with torch.no_grad():
        for batch_id, batch in enumerate(val_dataloader):
            inputs, labels = batch
            features = feature_extractor(inputs)

            for i, label_idx in enumerate(labels):
                label_name = cate_names[label_idx.item()]
                if label_name not in cate_features:
                    cate_features[label_name] = []
                cate_features[label_name].append(features[i].cpu())

    cate2feature = {}
    for name in cate_names:
        cate_feature = cate_features[name]
        cate_feature = numpy.array(cate_feature).squeeze()
        cate_feature = cate_feature.mean(axis=0)
        cate2feature[name] = cate_feature

    x = []
    y = []
    for cate_i in cate_names:
        for cate_j in cate_names:
            vec1 = cate2vec[cate_i]
            vec2 = cate2vec[cate_j]
            sim = vec1.dot(vec2) / (norm(vec1) * norm(vec2))
            x.append(sim)

            vec1 = cate2feature[cate_i]
            vec2 = cate2feature[cate_j]
            sim = vec1.dot(vec2) / (norm(vec1) * norm(vec2))
            y.append(sim)

    a = stats.pearsonr(np.array(x), np.array(y))
    return a[0]

def symbolicity_eval(messages, labels, epochs = 10, num_classes = 10):

    messages = torch.squeeze(messages)
    messages = messages.unsqueeze(1)
    messages = messages.repeat(1, 3, 1, 1)
    print(messages.shape)
    ids = labels

    train_messages, val_messages = np.split(messages, [int(len(messages) * 0.9)])
    train_labels, val_labels = np.split(ids, [int(len(ids) * 0.9)])

    train_sketches = SketchDataset(train_messages,train_labels)
    test_sketches = SketchDataset(val_messages,val_labels)

    train_indicies = np.random.choice(len(train_sketches), size=300, replace=False)
    train_sketches = Subset(train_sketches,train_indicies)

    test_indicies = np.random.choice(len(test_sketches), size=60, replace=False)
    test_sketches = Subset(test_sketches, test_indicies)

    train_dataloader = DataLoader(train_sketches, batch_size=16, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(test_sketches, batch_size=16, shuffle=False, num_workers=2)

    vgg = models.vgg16(pretrained=False)
    vgg.load_state_dict(torch.load("data/vgg16_pretrained.pth"))

    vgg.classifier[-1] = nn.Linear(vgg.classifier[-1].in_features, num_classes)

    for param in vgg.features.parameters():
        param.requires_grad = False

    for param in vgg.classifier[:-1].parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg.parameters(), lr=0.0002, momentum=0.9)


    for epoch in range(epochs):
        print("epoch:", epoch)
        running_loss = 0.0
        for batch_id, batch in enumerate(train_dataloader):
            inputs, labels = batch

            optimizer.zero_grad()

            outputs = vgg(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if batch_id % 2000 == 1999:  # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {batch_id + 1:5d}] loss: {running_loss / 2000:.3f}')
            #     running_loss = 0.0
        print(f'[{epoch + 1}] loss: {running_loss / len(train_dataloader):.3f}')

    print("evaluating symbolicity ...")

    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_id, batch in enumerate(val_dataloader):
            inputs, labels = batch
            outputs = vgg(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    eval_loss = running_loss / len(val_dataloader)
    eval_accuracy = (100 * correct / total)

    sem_col = semantic_correlaton(vgg, val_dataloader)

    return eval_loss , eval_accuracy, sem_col
