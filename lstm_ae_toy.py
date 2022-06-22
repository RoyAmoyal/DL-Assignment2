import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import koren_ae
import random_data
import matplotlib.pyplot as plt
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# hyper-params
epoch_num = 600
batch_size = 400
data_set = "random"
random_input_dim = 10000
random_seq_len = 50
random_latent_dim = 40
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_points(g_t, prediction):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
         31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    prediction = prediction.cpu()
    g_t = g_t.cpu()
    plt.plot(x, g_t, label='original signal')
    plt.plot(x, prediction, label='reconstruction')
    plt.legend()
    plt.ylabel('value')
    plt.xlabel('time step')
    plt.show()


if data_set == 'random':
    random_object = random_data.random_data(batch_size, random_input_dim, random_seq_len)
    trainloader = random_object.trainning_iterator()
    testloader = random_object.testing_iterator()
    validationloader = random_object.validation_iterator()
    model = koren_ae.koren_AE(random_seq_len, random_latent_dim)
elif data_set == 'MNIST': # pixel mnist
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])
    trainset = torchvision.datasets.MNIST(root='../data/', train=True, download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    model = koren_ae.koren_AE(28*28, 500)

model = model.double()
# opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
opt = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


# iterate over #epochs
for epoch in range(epoch_num):
    total_loss = 0.0
    # iterate over the dataset
    for i, data in enumerate(trainloader):
        if data_set != 'random':
            inputs, _ = data
            inputs = torch.squeeze(inputs, 1)
            inputs = torch.reshape(inputs, (inputs.shape[0], 28 * 28))
        else:
            inputs = data
        # inputs = torch.unsqueeze(inputs, 2) # changed here
        inputs = inputs.double()
        opt.zero_grad()
        inputs = inputs.unsqueeze(0)
        inputs = inputs.to(device)
        model.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0) # gradient clipping
        opt.step()


        # print stats
        total_loss += loss.item()
    print('[%d] loss: %.6f' % (epoch + 1, total_loss))

with torch.no_grad():
    total, correct = 0, 0
    for data in testloader:
        if data_set != 'random':
            inputs, _ = data
            inputs = torch.reshape(inputs, (inputs.shape[0], 28 * 28))
        else:
            inputs = data
            # inputs = torch.unsqueeze(inputs, 2)  # changed here
    inputs = torch.unsqueeze(inputs, 0)  # changed here

    inputs = inputs.to(device)
    model = model.to(device)
    outputs = model(inputs)
    print(outputs.shape)
    print(inputs.shape)
    plot_points(inputs[0][0], outputs[0][0])
    plot_points(inputs[0][1], outputs[0][1])
    time.sleep(2)
    plot_points(inputs[0][2], outputs[0][2])




