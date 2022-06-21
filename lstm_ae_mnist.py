import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import koren2_ae
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    pass

# hyper-params
epoch_num = 3
batch_size = 20


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])
trainset = torchvision.datasets.MNIST(root='../data/', train=True, download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

model = koren2_ae.koren_AE(28, 24)
model = model.double()
opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# opt = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()


def train():
    for epoch in range(epoch_num):
        total_loss = 0.0
        # iterate over the dataset
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = torch.squeeze(inputs)
            # inputs = torch.squeeze(inputs, 1)
            # inputs = torch.reshape(inputs, (inputs.shape[0], 28 * 28))
            inputs = inputs.double()
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)  # gradient clipping
            opt.step()
            # print stats
            total_loss += loss.item()
            if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, total_loss/100))
            total_loss = 0

train()

with torch.no_grad():
    total, correct = 0, 0
    for data in testloader:
        inputs, _ = data
        inputs = torch.squeeze(inputs)
        inputs = inputs.double()
        # inputs = torch.reshape(inputs, (inputs.shape[0], 28 * 28))
        outputs = model(inputs)
        imshow(torchvision.utils.make_grid(torch.unsqueeze(inputs, 1)))
        imshow(torchvision.utils.make_grid(torch.unsqueeze(outputs, 1)))
        exit()



