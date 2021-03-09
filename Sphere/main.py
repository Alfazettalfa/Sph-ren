import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Data import DatasetLoader
from matplotlib import pyplot as plt
import  numpy as np

#sdfsdfsdf
class NormalModel(nn.Module):
    def __init__(self):
        super(NormalModel, self).__init__()
        self.input = nn.Linear(20,23)
        self.hidden1 = nn.Linear(23,23)
        self.hidden2 = nn.Linear(23, 20)
        self.out = nn.Linear(20, 5)
        self.test_tensor = torch.tensor(torch.rand(1,5), requires_grad=True)
        print(self.test_tensor)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return torch.sigmoid(self.out(x))+self.test_tensor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.001
batch_size = 100
num_epochs = 50

dataset = DatasetLoader(transform=torch.from_numpy)
train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset)-2000, 2000])
train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)


model = NormalModel()
print(model)
s = 0
for p in model.parameters():
    P = 1
    for v in p.shape:
        P *= v
    s += P
print(s)
model = model.to(device=device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheluder = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=.5, patience=30)


def train():
    l = []
    for epoch in range(1, num_epochs+1):
        error = []
        target = torch.zeros(batch_size, dtype=torch.float32)
        score = torch.zeros([batch_size, 5], dtype=torch.float32)
        for index, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = model(data.float())
            target[index%batch_size] = targets.squeeze()
            score[index%batch_size] = scores
            if not index%(batch_size-1):
                loss = criterion(score, torch.tensor(target, dtype=torch.long))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                target = torch.zeros(batch_size, dtype=torch.float32)
                score = torch.zeros([batch_size, 5], dtype=torch.float32)
                error.append(loss.item())
                scheluder.step(sum(error))

                l.append(loss.item())
        print(model.test_tensor)
        check_accuracy(test_loader, model)
        plt.plot(l)
        plt.show(block=False)
        plt.pause(0.1)
        plt.cla()


def check_accuracy(loader, model):

    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for index, (data, targets) in enumerate(loader):
            num_samples += 1
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = model(data.float())
            if torch.argmax(scores.squeeze()) == targets.squeeze().item():
                num_correct += 1
    print(f"{num_correct} / {num_samples}, {num_correct/num_samples*100:.4f}%")
    model.train()




train()
check_accuracy(test_loader, model)
check_accuracy(train_loader, model)
torch.save(model.state_dict(), "Model.pt")
