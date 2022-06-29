import torch
import torch.nn as nn
import torch.nn.functional as F

#
class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_activ, out_activ):
        super(Encoder, self).__init__()
        self.num_layers = input_dim
        self.layers = nn.ModuleList()
        # for i in range(input_dim-out_dim):  # decreasing the dimension in 1 in each lstm til output dim
        #     layer = nn.LSTM(
        #         input_size=input_dim-i,
        #         hidden_size=input_dim-i-1,
        #         num_layers=1,
        #         batch_first=True
        #     )
        #     self.layers.append(layer)
        # for i in range(out_dim):
        #     layer = nn.LSTM(
        #         input_size=out_dim,
        #         hidden_size=out_dim,
        #         num_layers=1,
        #         batch_first=True
        #     )
        #     self.layers.append(layer)
        layer = nn.LSTM(
            input_size=input_dim,
            hidden_size=out_dim,
            num_layers=1,
            batch_first=True
        )
        self.layers.append(layer)

        self.h_activ, self.out_activ = h_activ, out_activ

    def forward(self, x):
        # x = x.unsqueeze(0)
        # exit()
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)
            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)
            elif self.out_activ and index == self.num_layers - 1:
                return self.out_activ(x).squeeze()
        return x.squeeze()


class Decoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_activ):
        super(Decoder, self).__init__()
        self.num_layers = input_dim
        self.layers = nn.ModuleList()
        # for i in range(out_dim-input_dim):
        #     layer = nn.LSTM(
        #         input_size=input_dim+i,
        #         hidden_size=input_dim+i+1,
        #         num_layers=1,
        #         batch_first=True
        #     )
        #     self.layers.append(layer)
        # for i in range(input_dim):
        #     layer = nn.LSTM(
        #         input_size=out_dim,
        #         hidden_size=out_dim,
        #         num_layers=1,
        #         batch_first=True
        #     )
        #     self.layers.append(layer)
        layer = nn.LSTM(
            input_size=input_dim,
            hidden_size=out_dim,
            num_layers=1,
            batch_first=True
        )
        self.layers.append(layer)

        self.h_activ = h_activ
        # self.dense_matrix = nn.Parameter(
        #     torch.rand((out_dim, out_dim), dtype=torch.float),
        #     requires_grad=True
        # )

    def forward(self, x, seq_len):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x, (h_n, c_n) = self.layers[0](x)  # only one block (of N layers)
        # for index, layer in enumerate(self.layers):
        #     x, (h_n, c_n) = layer(x)
        #
        #     if self.h_activ and index < self.num_layers - 1:
        #         x = self.h_activ(x)
        return x



class koren_AE(nn.Module):
    def __init__(self, input_dim, encoding_dim, classification=False, prediction=True, pix_by_pix = False, h_activ=nn.Sigmoid(),
                 out_activ=nn.Tanh()):
        super(koren_AE, self).__init__()
        self.encoding_dim = encoding_dim
        self.encoder = Encoder(input_dim, encoding_dim, h_activ, out_activ)
        self.decoder = Decoder(encoding_dim, input_dim, h_activ)
        self.conv1 = nn.Conv2d(1, 6, 5)   # maybe 5X5
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 4*4 are the remaining pixels. was 28X28 -> 24X24 -> 12X12 -> 8X8 -> 4X4
        self.fc2 = nn.Linear(120, 84)          # because according to the picture up, after every conv there is max-pooling
        self.fc3 = nn.Linear(84, 10)
        self.classify = classification
        self.pix_by_pix = pix_by_pix
        # self.linearDecoder = torch.nn.Linear(60, 1)
        self.prediction = prediction

    def forward(self, x, classification):
        seq_len = x.shape[0]
        x = self.encoder(x)
        if self.pix_by_pix == True:
            x = torch.reshape(x, (x.shape[0], 1, self.encoding_dim))
        x = self.decoder(x, seq_len)
        clone_x = x
        if classification:
            x = torch.reshape(x, (x.shape[0], 28, 28))

        if self.classify == True:
            x = x.unsqueeze(1)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return clone_x, x
        # x = self.linearDecoder(x)
        return x