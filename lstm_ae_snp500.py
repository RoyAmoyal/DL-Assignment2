import numpy as np
import pandas as pd
import warnings; warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import torch
import torch.nn as nn
import koren2_ae
import torch.optim as optim
from snp500dataset import sp500Dataset


def plot_points(g_t, prediction):
    x = [i+1 for i in range(1007)]
    prediction = prediction.cpu()
    g_t = g_t.cpu()
    plt.plot(x, g_t, label='original stock')
    plt.plot(x, prediction, label='reconstruction stock')
    plt.legend()
    plt.ylabel('value')
    plt.xlabel('day')
    plt.show()

stocks = pd.read_csv('SP500.csv', error_bad_lines=False)

def plotStocks():
    # plot daily max values for google and amazon during the years
    stocks = pd.read_csv('SP500.csv', error_bad_lines=False)
    fig, ax = plt.subplots(2, sharex=True, figsize=(16,6))
    amazon = stocks.loc[stocks['symbol'] == 'AMZN']
    google = stocks.loc[stocks['symbol'] == 'GOOGL']
    amazon.groupby('date')['high'].sum().plot(ax=ax[0])
    google.groupby('date')['high'].sum().plot(ax=ax[1])
    ax[0].set_title('Amazon stock during the years')
    ax[1].set_title('google stock during the years')
    plt.show()


epoch_num = 500
batch_size = 20
time_size = 1007
input_size = 1
seq_len = 53  # split the 1007 days into 19 * 53
latent_dim = 30
prediction = True
data_gen = sp500Dataset()
trainloader, validation_data = data_gen.getDataForModel(batch_size, time_size)
validationloader = validation_data.reshape(len(validation_data), input_size, time_size)



# hyper-params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using the device: ",device)



model = koren2_ae.koren_AE(seq_len, latent_dim, classification=False)
model = model.double()
opt = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


def prediction_criterion(criterion1, criterion2, outputs, inputs, labels_out, labels):
    lam1, lam2 = 1, 1
    return lam1 * criterion1(outputs, inputs) + lam2 * criterion2(labels_out, labels)


# iterate over #epochs
for epoch in range(epoch_num):
    total_loss = 0.0
    # iterate over the dataset
    for i, data in enumerate(trainloader):
        inputs = data
        # inputs = torch.unsqueeze(inputs, 2) # changed here
        inputs = torch.squeeze(inputs)
        inputs = inputs.double()
        inputs = torch.reshape(inputs, (inputs.shape[0], 19, 53))
        opt.zero_grad()
        # inputs = inputs.unsqueeze(0)
        inputs = inputs.to(device)
        model.to(device)
        if prediction == True:
            target = inputs[:, :, 1:53]
            inputs = inputs[:, : ,:52]
            outputs, label_out = model(inputs, classification=False, prediction=True) # we need to make the AE support prediction
            loss = prediction_criterion(criterion, criterion, outputs, inputs, label_out, target)
        else:
            outputs = model(inputs, False)
            loss = criterion(outputs, inputs)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0) # gradient clipping
        opt.step()
        # print stats
        total_loss += loss.item()
    print('[%d] loss: %.6f' % (epoch + 1, total_loss))

with torch.no_grad():
    total, correct = 0, 0
    counter = 0
    for data in validationloader:
        inputs = data
        inputs = torch.squeeze(inputs)
        inputs = inputs.double()
        inputs = inputs.to(device)
        inputs = torch.reshape(inputs, (1, 19, 53))

        outputs = model(inputs, False)
        outputs = torch.reshape(outputs, (1, 1, 1007))
        inputs = torch.reshape(inputs, (1, 1, 1007))
        outputs = outputs.squeeze()
        inputs = inputs.squeeze()
        plot_points(inputs, outputs)
        counter += 1
        if counter == 2:
            break

