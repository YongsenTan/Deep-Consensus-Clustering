import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if args.cuda else 'cpu')

        # AutoEncoder
        self.encoder = EncoderRNN(args)
        self.decoder = DecoderRNN(args)

        # Outcome Predictor
        self.predictor = Predictor(args)

        self.kmeans = None
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.MSELoss = nn.MSELoss()
        self.BCELoss = nn.BCELoss()

        self.to(self.device)

    # Get encoded data , reconstruct data and outcome prediction
    def forward(self, x):
        encoded_x, decoded_x = self.get_rep(x)
        encoded_x = encoded_x[:, 0, :]
        output_outcome = self.predictor(encoded_x)
        return encoded_x, decoded_x, output_outcome.view(-1)

    # Get encoded representation and reconstruct data
    def get_rep(self, x):
        encoded_x, (hn, cn), new_input = self.encoder(x)
        decoded_x = self.decoder(new_input, (hn, cn))
        return encoded_x, decoded_x

    # Pretrain the autoencoder
    def pretrain(self, dataloader_train, verbose=True):
        if verbose:
            print('========== Start pretraining ==========')

        rec_loss_list = []

        self.train()
        for epoch in range(self.args.pre_epoch):
            error = []
            for batch_idx, (data_x, target) in enumerate(dataloader_train):
                data_x = data_x.to(self.device)
                enc, dec = self.get_rep(data_x)
                self.optimizer.zero_grad()
                loss = self.MSELoss(data_x, dec)
                loss.backward()
                self.optimizer.step()
                error.append(loss.detach().cpu().numpy())
            rec_loss = np.mean(error)
            if verbose:
                print("Epoch: %s | train AE loss: %.3f" % (epoch, float(rec_loss)))
            rec_loss_list.append(rec_loss)

        if verbose:
            print('=========== End pretraining ===========\n')

        return rec_loss_list

    # Fit the dataset
    def fit(self, dataloader_train, verbose=True):
        error_ae = []
        error_outcome = []
        y = []
        out = []

        self.train()
        for batch_idx, (data_x, target) in enumerate(dataloader_train):
            data_x = data_x.to(self.device)
            target = target.to(self.device)

            encoded_x, decoded_x, output_outcome = self(data_x)

            self.optimizer.zero_grad()

            loss_ae = self.MSELoss(data_x, decoded_x)
            loss_outcome = self.BCELoss(output_outcome, target)
            loss = self.args.lambda_AE * loss_ae + self.args.lambda_outcome * loss_outcome
            loss.backward()

            self.optimizer.step()

            error_ae.append(loss_ae.data.cpu().numpy())
            error_outcome.append(loss_outcome.data.cpu().numpy())

            y.append(target.data.cpu())
            out.append(output_outcome.data.cpu())

        train_ae_loss = np.mean(error_ae)
        train_outcome_loss = np.mean(error_outcome)
        train_outcome_auc_score = roc_auc_score(np.concatenate(y, 0),
                                                np.concatenate(out, 0),
                                                multi_class='ovr')
        if verbose:
            print('ae_loss: %.3f' % train_ae_loss)
            print('outcome_loss: %.3f' % train_outcome_loss)
            print('outcome_auc_score: %.3f' % train_outcome_auc_score)
        return train_ae_loss, train_outcome_loss, train_outcome_auc_score


class EncoderRNN(nn.Module):
    def __init__(self, args):
        super(EncoderRNN, self).__init__()
        # Hidden size of LSTM
        self.hidden_dims = args.hidden_dims
        # Dimension of input features
        self.input_dim = args.input_dim
        # Number of LSTM layers
        self.n_layers = args.n_layers
        self.dropout = args.dropout
        self.cuda = args.cuda
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dims,
                            num_layers=self.n_layers,
                            dropout=self.dropout,
                            bidirectional=True,
                            batch_first=True)
        self.init_weights()

    # Uniform initialize parameters
    def init_weights(self):
        for p in self.lstm.parameters():
            p.data.uniform_(-0.1, 0.1)

    # Get representation of data and cell state & input of Decoder
    def forward(self, x):
        # output [batch_size, seq_size, hidden_size * 2]
        # hn & cn [directions(1 or 2), batch_size, hidden_size]
        output, (hn, cn) = self.lstm(x)

        # Flip along the timestamp dimension
        output = torch.flip(output, [1])
        new_input = torch.flip(x, [1])

        # zeros [batch_size, 1, input_size]
        zeros = torch.zeros(x.shape[0], 1, x.shape[-1])
        if self.cuda:
            zeros = zeros.cuda()

        # concat along the timestamp dimension
        new_input = torch.cat((zeros, new_input), 1)

        # drop the last timestamp
        # new_input [batch_size, seq_size, input_size]
        new_input = new_input[:, :-1, :]
        return output, (hn, cn), new_input


class DecoderRNN(nn.Module):
    def __init__(self, args):
        super(DecoderRNN, self).__init__()
        self.hidden_dims = args.hidden_dims
        self.input_dim = args.input_dim
        self.n_layers = args.n_layers
        self.dropout = args.dropout
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dims,
                            num_layers=self.n_layers,
                            dropout=self.dropout,
                            bidirectional=True,
                            batch_first=True)
        # Bidirectional
        self.linear_decoder_output = nn.Linear(args.hidden_dims * 2, args.input_dim)
        self.init_weights()

    def forward(self, x, h):
        # output [batch_size, seq_size, input_size]
        output, _ = self.lstm(x, h)

        # Flip along the timestamp dimension
        output = torch.flip(output, [1])

        output = self.linear_decoder_output(output)
        return output

    # Uniform initialize parameters
    def init_weights(self):
        for p in self.lstm.parameters():
            p.data.uniform_(-0.1, 0.1)
        self.linear_decoder_output.bias.data.fill_(0)
        self.linear_decoder_output.weight.data.uniform_(-0.1, 0.1)


class Predictor(nn.Module):
    def __init__(self, args):
        super(Predictor, self).__init__()
        # Bidirectional
        self.linear_1 = nn.Linear(args.hidden_dims * 2, args.hidden_dims * 2)
        self.Softmax = nn.Softmax(dim=1)
        self.linear_2 = nn.Linear(args.hidden_dims * 2, 1)
        self.Sigmoid = nn.Sigmoid()
        self.init_weights()

    def forward(self, encoded_x):
        # x  [batch_size, n_clusters]
        x = self.linear_1(encoded_x)
        x = self.Softmax(x)
        x = self.linear_2(x)
        x = self.Sigmoid(x)
        return x.squeeze()

    # Uniform initialize parameters
    def init_weights(self):
        self.linear_1.bias.data.fill_(0)
        self.linear_1.weight.data.uniform_(-0.1, 0.1)

        self.linear_2.bias.data.fill_(0)
        self.linear_2.weight.data.uniform_(-0.1, 0.1)
