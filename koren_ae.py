import torch
import torch.nn as nn

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
        x, (h_n, c_n) = self.layers[0](x)  # only one block (of N layers)
        # for index, layer in enumerate(self.layers):
        #     x, (h_n, c_n) = layer(x)
        #
        #     if self.h_activ and index < self.num_layers - 1:
        #         x = self.h_activ(x)
        return x



class koren_AE(nn.Module):
    def __init__(self, input_dim, encoding_dim, h_activ=nn.Sigmoid(),
                 out_activ=nn.Tanh()):
        super(koren_AE, self).__init__()
        self.encoder = Encoder(input_dim, encoding_dim, h_activ, out_activ)
        self.decoder = Decoder(encoding_dim, input_dim, h_activ)
        # self.linearDecoder = torch.nn.Linear(60, 1)

    def forward(self, x):
        seq_len = x.shape[0]
        x = self.encoder(x)
        x = self.decoder(x, seq_len)
        # x = self.linearDecoder(x)
        return x