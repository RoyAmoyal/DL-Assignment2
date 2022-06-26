import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import koren2_ae
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def imshow(img,some_string):
    img = img.cpu()
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.title(some_string)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(some_string)
    plt.show()
    pass

# hyper-params
epoch_num = 30
batch_size = 200
classification = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using the device: ",device)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])
trainset = torchvision.datasets.MNIST(root='../data/', train=True, download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False)


classes = ('0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9')


model = koren2_ae.koren_AE(28, 15, classification)
model = model.double()
model = model.to(device)
# opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
opt = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()

def classification_criterion(criterion1, criterion2, outputs, inputs, labels_out, labels):
    lam1, lam2 = 1, 1
    return lam1 * criterion1(outputs, inputs) + lam2 * criterion2(labels_out, labels)


for epoch in range(epoch_num):
    total_loss = 0.0
    # iterate over the dataset
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = torch.squeeze(inputs)
        # inputs = torch.squeeze(inputs, 1)
        # inputs = torch.reshape(inputs, (inputs.shape[0], 28 * 28))
        inputs = inputs.double()
        inputs = inputs.to(device)
        opt.zero_grad()
        if classification == False:
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
        else:
            outputs, label_out = model(inputs, classification)
            labels = labels.to(device)
            loss = classification_criterion(criterion, criterion2, outputs, inputs, label_out, labels)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)  # gradient clipping
        opt.step()
        # print stats
        total_loss += loss.item()
        if i % 100 == 99:
                print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, total_loss/100))
                total_loss = 0
# =======
# for epoch in range(epoch_num):
#     total_loss = 0.0
#     # iterate over the dataset
#     for i, data in enumerate(trainloader):
#         inputs, labels = data
#         inputs = torch.squeeze(inputs)
#         # inputs = torch.squeeze(inputs, 1)
#         # inputs = torch.reshape(inputs, (inputs.shape[0], 28 * 28))
#         inputs = inputs.double()
#         opt.zero_grad()
#         inputs = inputs.to(device)
#         model = model.to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs, inputs)
#         loss.backward()
#         nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)  # gradient clipping
#         opt.step()
#         # print stats
#         total_loss += loss.item()
#     print('[%d] loss: %.7f' % (epoch + 1, total_loss))
#     total_loss = 0
# >>>>>>> gpu_branch



with torch.no_grad():
    total, correct = 0, 0
    for data in testloader:
        inputs, labels = data
        inputs = torch.squeeze(inputs)
        inputs = inputs.double()
        inputs = inputs.to(device)

        # inputs = torch.reshape(inputs, (inputs.shape[0], 28 * 28))
        if classification == False: # then show me 2 examples, nothing to test
            outputs = model(inputs)
            imshow(torchvision.utils.make_grid(torch.unsqueeze(inputs, 1)), "Original")
            imshow(torchvision.utils.make_grid(torch.unsqueeze(outputs, 1)), "Reconstructed")
        else:
            outputs, label_out = model(inputs, classification)
            labels = labels.to(device)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(label_out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))



