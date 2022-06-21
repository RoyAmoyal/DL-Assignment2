import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import koren_ae
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    pass

# hyper-params
epoch_num = 1
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

model = koren_ae.koren_AE(28, 15)
model = model.double()
# opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
opt = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()


def train():
    for epoch in range(epoch_num):
        total_loss = 0.0
        # iterate over the dataset
        for i, data in enumerate(trainloader):
            inputs, _ = data
            # inputs = torch.squeeze(inputs, 1)
            # inputs = torch.reshape(inputs, (inputs.shape[0], 28 * 28))
            inputs = inputs.double()
            for j in range(28):
                temp_input = inputs[:, :, :, j]
                temp_input = torch.squeeze(temp_input, 1)
                opt.zero_grad()
                outputs = model(temp_input)
                loss = criterion(outputs, temp_input)
                loss.backward()
                opt.step()
                # print stats
                total_loss += loss.item()
                if i % 5 == 4:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, total_loss/5))
            if i == 20:
                break
                total_loss = 0

train()

with torch.no_grad():
    total, correct = 0, 0
    for data in testloader:
        inputs, _ = data
        inputs = torch.squeeze(inputs, 1)
        inputs = inputs.double()
        # inputs = torch.reshape(inputs, (inputs.shape[0], 28 * 28))
        # outputs = model(inputs)
        imshow(torchvision.utils.make_grid(inputs))
        outputs_pic = inputs
        for j in range(28):
            temp_input = inputs[:, :, :, j]
            temp_input = torch.squeeze(temp_input, 1)
            outputs = model(temp_input)
            unsq_outputs = torch.unsqueeze(outputs, 1)
            outputs_pic[:, :, :, j] = unsq_outputs
        imshow(torchvision.utils.make_grid(outputs_pic))



