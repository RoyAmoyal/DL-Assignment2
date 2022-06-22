import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import LSTM_AE
import koren_ae
import random_data


if __name__ == '__main__':
    # hyper-params
    epoch_num = 10
    batch_size = 5
    data_set = "random"
    random_input_dim = 10000
    random_seq_len = 50
    random_latent_dim = 30
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if data_set == 'random':
        trainloader = random_data.random_data(batch_size, random_input_dim, random_seq_len).iterator()
        model = koren_ae.koren_AE(random_seq_len, random_latent_dim)
    elif data_set == 'MNIST':  # pixel mnist
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(0.5, 0.5)])
        trainset = torchvision.datasets.MNIST(root='../data/', train=True, download=True,
                                                transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        model = koren_ae.koren_AE(28*28, 500)

    model = model.double()
    # opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    opt = optim.Adam(model.parameters(), lr=1e-3)
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
            inputs = inputs.double()
            opt.zero_grad()
            # prepare data
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            opt.step()

            # print stats
            total_loss += loss.item()
            if i % 2 == 1:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, total_loss / 2))
            total_loss = 0

    with torch.no_grad():
        total, correct = 0, 0
        for data in trainloader:
            if data_set != 'random':
                inputs, _ = data
                inputs = torch.squeeze(inputs, 1)
                inputs = torch.reshape(inputs, (inputs.shape[0], 28 * 28))
            else:
                inputs = data
            inputs = inputs.to(device)
            outputs = model(inputs)

            print("the input:", inputs)
            print("the output:", outputs)



