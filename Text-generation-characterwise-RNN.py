#%%
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from datetime import datetime as dt

#%%
with open(os.getcwd() + '/data/anna.txt') as f:
    text = f.read()

#%%

# generating integer coding for characters
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {char: ii for ii, char in int2char.items()}

# encoding whole text
encoded = np.array([char2int[char] for char in text])


#%%
def one_hot_encoding(arr, n_labels):
    """
    :param arr: numpy array containing elements encoded to integers
    :param n_labels: total size of dictionary
    :return: one-hot encoded array
    """

    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)  # initialize properly sized array with 0s
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1  # fill appropriate positions with 1s
    one_hot = one_hot.reshape((*arr.shape, n_labels))  # reshape array to original dimensions

    return one_hot


#%%
array_ex = np.array([[3, 5, 1]])
num_labels = 8
one_hot_encoding(array_ex, num_labels)


#%%
def get_batches(arr, batch_size, seq_length):
    """
    :param arr: numpy array containing elements encoded to integers
    :param batch_size: number of data sequences in one batch
    :param seq_length: length of sequence in batches
    :return: one batch per function call (generator) for training data (x) and target (y)
    """
    n_batches = arr.shape[0] // (batch_size * seq_length)
    arr = arr[:batch_size * n_batches * seq_length]
    arr = arr.reshape((batch_size, -1))

    for n in range(0, arr.shape[1], seq_length):
        x = arr[:, n:n+seq_length]
        y = np.zeros_like(x)  # this is to avoid y going over the boundaries of arr at the last batch
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


#%%
batches = get_batches(encoded, 8, 50)
x, y = next(batches)
print('x\n', x)
print('y\n', y)

#%%
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Training on GPU')
else:
    print('No GPU available; training on CPU. Expect LONG runtimes - or keep the number of epochs limited.')


#%%
class CharRNN(nn.Module):

    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {char: ii for ii, char in self.int2char.items()}

        self.lstm = nn.LSTM(len(self.chars), self.n_hidden,
                            self.n_layers, dropout=self.drop_prob, batch_first=True)
        # here dropout automatically creates dropout layer between LSTM cells

        self.dropout = nn.Dropout(p=self.drop_prob)  # we need this for dropout bw LSTM and final FC layer

        self.fc = nn.Linear(in_features=self.n_hidden, out_features=len(self.chars))

    def forward(self, x, hidden):
        """
        Forward pass through the network.
        :param x: input sequence of characters (one-hot encoded)
        :param hidden: values of hidden layer
        :return: final output, hidden state
        """
        lstm_out, hidden_state = self.lstm(x, hidden)
        out = self.dropout(lstm_out)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden_state

    def init_hidden(self, batch_size):
        """
        Initializes hidden state.
        :param batch_size: batch size
        :return: hidden state
        """
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden


#%%
def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    """
    Defines the process for training the network
    :param net: CharRNN network
    :param data: text data to train the network
    :param epochs: number of epochs to run
    :param batch_size: number of data records in a batch
    :param seq_length: number of characters in one line of a mini-batch
    :param lr: learning rate
    :param clip: limit for gradient clipping
    :param val_frac: fraction of validation set compared to total size of input data
    :param print_every: number of steps by which loss is printed to console
    :return: none
    """

    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if train_on_gpu:
        net.cuda()

    counter = 0
    n_chars = len(net.chars)

    for e in range(epochs):
        h = net.init_hidden(batch_size)

        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1

            x = one_hot_encoding(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            h = tuple([each.data for each in h])

            net.zero_grad()

            output, h = net(inputs, h)

            loss = criterion(output, targets.contiguous().view(batch_size*seq_length))
            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), clip)

            opt.step()

            if counter % print_every == 0:
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    x = one_hot_encoding(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if train_on_gpu:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)

                    val_loss = criterion(output, targets.contiguous().view(batch_size*seq_length))

                    val_losses.append(val_loss.item())

                net.train()

                print('Epoch: {}/{}...'.format(e+1, epochs),
                      'Step: {}...'.format(counter),
                      'Loss: {:.4f}...'.format(loss.item()),
                      'Val loss: {:.4f}...'.format(np.mean(val_losses)))


#%%
n_hidden = 512
n_layers = 2

net = CharRNN(tokens=chars, n_hidden=n_hidden, n_layers=n_layers)
print(net)
for i in net.named_parameters():
    print(i)

#%%
batch_size = 128
seq_length = 100
n_epochs = 2

train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)

#%%
model_name = 'textgen-char-rnn_{}_{}.net'.format(dt.today().date(), int(dt.today().timestamp()))

checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}

with open(f'./{model_name}', 'wb') as f:
    torch.save(checkpoint, f)


#%%
def predict(net, char, h=None, top_k=None):
    """
    Implements the prediction procedure for the character-level RNN
    :param net: trained RNN for character-level prediction
    :param char: character for which to predict the next character
    :param h:
    :param top_k:
    :return:
    """

    # tensor inputs
    x = np.array([[net.char2int[char]]])
    x = one_hot_encoding(x, len(net.chars))
    inputs = torch.from_numpy(x)

    if train_on_gpu:
        inputs = inputs.cuda()

    # detach hidden state from history
    h = tuple([each.data for each in h])

    out, h = net(inputs, h)

    # get character probs
    p = F.softmax(out, dim=1).data
    if train_on_gpu:
        p = p.cpu()  # we're transferring p to CPU for further calculations

    # get top k characters
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())

    return net.int2char[char], h


#%%
def sample(net, size, prime='The', top_k=None):

    if train_on_gpu:
        net.cuda()
    else:
        net.cpu()

    net.eval()

    chars = [ch for ch in prime]
    h = net.init_hidden(1)

    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)

    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)


#%%
# load checkpoint from saved model
with open(os.getcwd() + '/rnn_20_epoch.net', 'rb') as f:
    checkpoint = torch.load(f, map_location='cpu')

loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])

#%%
print(sample(loaded, 2000, top_k=5, prime='And Levin said'))
