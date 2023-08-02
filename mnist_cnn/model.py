import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(800, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1) 
        x = F.softmax(self.fc1(x), dim=1)
        return x

class MNIST_Model():

    # Init Network Class
    def __init__(self, learning_rate=4.12e-4, batch_size=32, epochs=20):
        self.net = Net()
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
    
    # Train Function
    def train_network(self):
        
        ll = nn.CrossEntropyLoss()
        oo = optim.Adam(self.net.parameters(), lr=self.lr)
        
        trainset = datasets.MNIST('./train', download=True, train=True, transform=self.transform)
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):

            running_loss = 0.0
            for i, (images, labels) in enumerate(trainloader):
                oo.zero_grad()
                outputs = self.net(images)
                loss = ll(outputs, labels)
                loss.backward()
                oo.step()
                running_loss += loss.item()

            print(f'[{epoch + 1}] loss: {running_loss / i}')

        print('Finished Training')

    # Predict from network
    def test_network(self):
        valset = datasets.MNIST('./test', download=True, train=False, transform=self.transform)
        valloader = DataLoader(valset, batch_size=len(valset), shuffle=True)
        images, labels = next(iter(valloader))
        res = torch.argmax(self.net(images), dim=1)
        print(f"Sample Results: {res[:10]}")
        print(f"Sample Labels: {labels[:10]}")
        print(f"Accuracy: {torch.sum(torch.eq(res, labels))/len(valset)}")
    
    # Save network to disc
    def save(self, path):
        torch.save(self.net.state_dict(), path)
    
    # Load network from disc
    def load(self, path):
        self.net = Net()
        self.net.load_state_dict(torch.load(path))

