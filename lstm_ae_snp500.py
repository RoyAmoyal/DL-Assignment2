import numpy as np
import pandas as pd
import warnings

warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms

# sns.set()
import torch
import torch.nn as nn
import koren3_ae
import koren2_ae
import torch.optim as optim
from snp500dataset import sp500Dataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def plot_points(g_t, prediction):
    x = [i for i in range(1007)]
    prediction = prediction.cpu()
    g_t = g_t.cpu()
    plt.plot(x, g_t, label='original stock')
    plt.plot(x, prediction, label='reconstruction stock')
    plt.legend()
    plt.ylabel('value')
    plt.xlabel('day')
    plt.show()


def plot_points2(g_t, prediction):
    x = [i for i in range(19)]
    prediction = prediction.cpu()[-19:]
    g_t = g_t.cpu()[-19:]
    plt.plot(x, g_t, label='original stock')
    plt.plot(x, prediction, label='reconstruction stock')
    plt.legend()
    plt.ylabel('value')
    ax = plt.gca()
    ax.set_ylim([0, 2])
    plt.xlabel('day')
    plt.show()


def plot_points3(g_t, prediction):
    x = [i for i in range(27)]
    plt.plot(x, g_t, label='true')
    plt.plot(x, prediction, label='prediction')
    plt.legend()
    plt.ylabel('value')
    plt.xlabel('day')
    plt.show()


def plot_losses(train_loss, pred_loss):
    x = [i for i in range(len(train_loss))]
    plt.plot(x, train_loss, label='Train Loss')
    plt.plot(x, pred_loss, label='Prediction Loss')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


stocks = pd.read_csv('SP500.csv', error_bad_lines=False)


def plotStocks():
    # plot daily max values for google and amazon during the years
    stocks = pd.read_csv('SP500.csv', error_bad_lines=False)
    fig, ax = plt.subplots(2, sharex=True, figsize=(16, 6))
    amazon = stocks.loc[stocks['symbol'] == 'AMZN']
    google = stocks.loc[stocks['symbol'] == 'GOOGL']
    amazon.groupby('date')['high'].sum().plot(ax=ax[0])
    google.groupby('date')['high'].sum().plot(ax=ax[1])
    ax[0].set_title('Amazon stock during the years')
    ax[1].set_title('google stock during the years')
    plt.show()


epoch_num = 2
batch_size = 20
time_size = 1007
input_size = 1
t = 26
prediction = True
question3dot3 = True
data_gen = sp500Dataset()
trainloader, validation_data = data_gen.getDataForModel(batch_size, time_size)
validationloader = validation_data.reshape(len(validation_data), input_size, time_size)

# hyper-params
begin = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using the device: ",device)

if prediction == False:
    seq_len = 53
    latent_dim = 30
    model = koren2_ae.koren_AE(seq_len, latent_dim, classification=False)
elif prediction == True and question3dot3 == True:
    seq_len = 52
    latent_dim = 30
    model = koren3_ae.koren_AE(seq_len, latent_dim, classification=False)
else:  # gimel
    t = 26
    seq_len = t
    latent_dim = 30
    model = koren3_ae.koren_AE(seq_len, latent_dim, classification=False)

model = model.double()
opt = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
training_loss = []
prediction_loss = []


def prediction_criterion(criterion1, outputs, inputs, pred, gt_pred):
    lam1, lam2 = 1, 1
    train_loss = criterion1(outputs, inputs)
    pred_loss = criterion1(pred, gt_pred)

    return lam1 * criterion1(outputs, inputs) + lam2 * criterion1(pred, gt_pred), train_loss, pred_loss


# iterate over #epochs
for epoch in range(epoch_num):
    total_loss = 0.0
    train_toal_loss =0.0
    pred_toal_loss =0.0
    pred_loss = None
    train_loss =None
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
        loss = None
        if prediction == False:
            target = inputs[:, :, :53]
            inputs = inputs[:, :, :52]
            outputs, label_out = model(inputs, classification=False)  # we need to make the AE support prediction
            loss = prediction_criterion(criterion, outputs, inputs, label_out, target)
            loss.backward()
        elif question3dot3 == True:
            target = inputs[:, :, 52]
            inputs = inputs[:, :, :52]
            outputs = model(inputs, False)
            loss, train_loss, pred_loss = prediction_criterion(criterion, outputs[:, :, :52], inputs, outputs[:, :, 52],
                                                               target)
            loss.backward()

        else:
            original = inputs
            while t != 52:

                target = original[:, :, t]  # 27 bindex 26
                inputs = original[:, :, begin:t]  # 26 items mindex 0 ad 25

                outputs = model(inputs, False)
                loss, train_loss, pred_loss = prediction_criterion(criterion, outputs[:, :, :26], inputs,
                                                                   outputs[:, :, 26],
                                                                   target)
                t += 1
                begin += 1
                loss.backward()
            t = 26
            begin = 0

        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)  # gradient clipping
        opt.step()
        # print stats
        total_loss += loss.item()
        train_toal_loss += train_loss
        pred_toal_loss += pred_loss

    training_loss.append(train_loss.cpu().detach().numpy())
    prediction_loss.append(pred_loss.cpu().detach().numpy())

    print('[%d] loss: %.6f' % (epoch + 1, total_loss))

with torch.no_grad():

    total, correct = 0, 0
    counter = 0
    for data in validationloader:
        inputs = data
        inputs = torch.squeeze(inputs)
        inputs = inputs.double()
        inputs = inputs.to(device)
        print(inputs.shape)
        inputs = torch.reshape(inputs, (1, 19, 53))
        print(inputs[:, :, :52].shape)
        outputs = model(inputs[:, :, :52], False)
        outputs = torch.reshape(outputs, (1, 1, 1007))
        inputs = torch.reshape(inputs, (1, 1, 1007))
        outputs = outputs.squeeze()
        inputs = inputs.squeeze()
        plot_points(inputs, outputs)
        plot_points2(inputs, outputs)

        counter += 1
        if counter == 3:
            break
    plot_losses(training_loss, prediction_loss)
