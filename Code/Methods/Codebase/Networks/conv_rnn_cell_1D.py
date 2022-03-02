#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter

torch_activations = {
    "celu": nn.CELU(),
    "selu": nn.SELU(),
    "tanh": nn.Tanh(),
    "identity": nn.Identity(),
}


def getActivation(str_):
    return torch_activations[str_]


class ConvRNNCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size,
                 activation="tanh",
                 cell_type="lstm",
                 torch_dtype=torch.DoubleTensor,
                 shape=(64)):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.rnn_cell_type = cell_type
        self.activation = getActivation(activation)
        self.torch_dtype = torch_dtype

        self.padding = int((self.kernel_size - 1) / 2)

        if self.rnn_cell_type == "lstm":

            self.Wxi = nn.Conv1d(self.input_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Wxf = nn.Conv1d(self.input_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Wxc = nn.Conv1d(self.input_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Wxo = nn.Conv1d(self.input_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)

            self.Whi = nn.Conv1d(self.hidden_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Whf = nn.Conv1d(self.hidden_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Whc = nn.Conv1d(self.hidden_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Who = nn.Conv1d(self.hidden_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)

        elif self.rnn_cell_type == "lstm_2":

            self.Wxi = nn.Conv1d(self.input_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Wxf = nn.Conv1d(self.input_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Wxc = nn.Conv1d(self.input_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Wxo = nn.Conv1d(self.input_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)

            self.Whi = nn.Conv1d(self.hidden_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Whf = nn.Conv1d(self.hidden_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Whc = nn.Conv1d(self.hidden_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Who = nn.Conv1d(self.hidden_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)

            self.Bci = Parameter(self.torch_dtype(1, hidden_channels,
                                                  shape[0]))
            self.Bcf = Parameter(self.torch_dtype(1, hidden_channels,
                                                  shape[0]))
            self.Bco = Parameter(self.torch_dtype(1, hidden_channels,
                                                  shape[0]))
            self.Bcc = Parameter(self.torch_dtype(1, hidden_channels,
                                                  shape[0]))

        elif self.rnn_cell_type == "lstm_3":

            self.Wxi = nn.Conv1d(self.input_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Wxf = nn.Conv1d(self.input_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Wxc = nn.Conv1d(self.input_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Wxo = nn.Conv1d(self.input_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)

            self.Whi = nn.Conv1d(self.hidden_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Whf = nn.Conv1d(self.hidden_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Whc = nn.Conv1d(self.hidden_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Who = nn.Conv1d(self.hidden_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)

            self.Wci = Parameter(self.torch_dtype(1, hidden_channels,
                                                  shape[0]))
            self.Wcf = Parameter(self.torch_dtype(1, hidden_channels,
                                                  shape[0]))
            self.Wco = Parameter(self.torch_dtype(1, hidden_channels,
                                                  shape[0]))

            self.Bci = Parameter(self.torch_dtype(1, hidden_channels,
                                                  shape[0]))
            self.Bcf = Parameter(self.torch_dtype(1, hidden_channels,
                                                  shape[0]))
            self.Bco = Parameter(self.torch_dtype(1, hidden_channels,
                                                  shape[0]))
            self.Bcc = Parameter(self.torch_dtype(1, hidden_channels,
                                                  shape[0]))

        elif self.rnn_cell_type == "gru":
            self.Wiz = nn.Conv1d(self.input_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Whz = nn.Conv1d(self.hidden_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=False)

            self.Wir = nn.Conv1d(self.input_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Whr = nn.Conv1d(self.hidden_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=False)

            self.Wih = nn.Conv1d(self.input_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=True)
            self.Whh = nn.Conv1d(self.hidden_channels,
                                 self.hidden_channels,
                                 self.kernel_size,
                                 stride=1,
                                 padding=self.padding,
                                 bias=False)

        else:
            raise ValueError("Not implemented.")
        """ Changing the type of the network modules """
        if (self.torch_dtype
                == torch.DoubleTensor) or (self.torch_dtype
                                           == torch.cuda.DoubleTensor):
            self.double()
        else:
            self.float()

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x, prev_state, is_train=True):
        with torch.set_grad_enabled(is_train):
            # print(x.size())
            # x = pad_circular(x, self.padding)
            # print(x.size())
            # print(ark)
            if self.rnn_cell_type == "lstm":
                h_prev = prev_state[0]
                c_prev = prev_state[1]

                # Input gate
                gi = torch.sigmoid(self.Wxi(x) + self.Whi(h_prev))
                # Forget gate
                gf = torch.sigmoid(self.Wxf(x) + self.Whf(h_prev))
                # Output gate
                go = torch.sigmoid(self.Wxo(x) + self.Who(h_prev))
                # Cell gate
                gc = self.activation(self.Wxc(x) + self.Whc(h_prev))

                c_next = (gf * c_prev) + (gi * gc)

                h_next = go * self.activation(c_next)

                next_hidden = torch.stack([h_next, c_next])
                output = h_next

            elif self.rnn_cell_type == "lstm_2":
                h_prev = prev_state[0]
                c_prev = prev_state[1]

                # Input gate
                gi = torch.sigmoid(self.Wxi(x) + self.Whi(h_prev) + self.Bci)
                # Forget gate
                gf = torch.sigmoid(self.Wxf(x) + self.Whf(h_prev) + self.Bcf)

                # Cell gate
                gc = self.activation(self.Wxc(x) + self.Whc(h_prev) + self.Bcc)

                c_next = (gf * c_prev) + (gi * gc)

                # Output gate
                go = torch.sigmoid(self.Wxo(x) + self.Who(h_prev) + self.Bco)

                h_next = go * self.activation(c_next)

                next_hidden = torch.stack([h_next, c_next])
                output = h_next

            elif self.rnn_cell_type == "lstm_3":
                h_prev = prev_state[0]
                c_prev = prev_state[1]

                # Input gate
                gi = torch.sigmoid(
                    self.Wxi(x) + self.Whi(h_prev) + c_prev * self.Wci +
                    self.Bci)
                # Forget gate
                gf = torch.sigmoid(
                    self.Wxf(x) + self.Whf(h_prev) + c_prev * self.Wcf +
                    self.Bcf)

                # Cell gate
                gc = self.activation(self.Wxc(x) + self.Whc(h_prev) + self.Bcc)

                c_next = (gf * c_prev) + (gi * gc)

                # Output gate
                go = torch.sigmoid(
                    self.Wxo(x) + self.Who(h_prev) + c_next * self.Wco +
                    self.Bco)

                h_next = go * self.activation(c_next)

                next_hidden = torch.stack([h_next, c_next])
                output = h_next

            elif self.rnn_cell_type == "gru":
                h_prev = prev_state
                z = torch.sigmoid(self.Wiz(x) + self.Whz(h_prev))
                r = torch.sigmoid(self.Wir(x) + self.Whr(h_prev))
                h_next = self.activation(self.Wih(x) + self.Whh(r * h_prev))
                output = (1. - z) * h_prev + z * h_next
                next_hidden = output
        return output, next_hidden

    def initializeHiddenState(self, input_batch):
        batch_size = input_batch.data.size()[0]
        spatial_size = input_batch.data.size()[2:]
        state_size = [batch_size, self.hidden_channels] + list(spatial_size)
        initial_state = (Variable(torch.zeros(state_size)),
                         Variable(torch.zeros(state_size)))
        return initial_state


def _main():
    """
    Run some basic tests on the API
    """

    # define batch_size, channels, width
    b, c, w = 32, 3, 16
    d = 5  # hidden state size
    lr = 1e-1  # learning rate
    T = 6  # sequence length
    max_epoch = 20  # number of epochs

    # set manual seed
    torch.manual_seed(0)

    input_channels = c
    hidden_channels = d
    kernel_size = 5

    print('Instantiate model')
    model = ConvRNNCell(input_channels, hidden_channels, kernel_size)
    print(repr(model))

    print('Create input and target Variables')
    x = Variable(torch.rand(T, b, c, w))
    y = Variable(torch.randn(T, b, d, w))

    print('Create a MSE criterion')
    loss_fn = nn.MSELoss()

    print('Run for', max_epoch, 'iterations')
    for epoch in range(0, max_epoch):
        state = model.initializeHiddenState(x[0])
        loss = 0
        for t in range(0, T):
            print("##")
            print(x[t].size())
            print(state[0].size())
            print(state[1].size())
            output, state = model(x[t], state)
            loss += loss_fn(output, y[t])

        print(' > Epoch {:2d} loss: {:.3f}'.format((epoch + 1), loss.data))

        # zero grad parameters
        model.zero_grad()

        # compute new grad parameters through time!
        loss.backward()

        # learning_rate step against the gradient
        for p in model.parameters():
            p.data.sub_(p.grad.data * lr)

    print('Input size:', list(x.data.size()))
    print('Target size:', list(y.data.size()))
    print('Last hidden state size:', list(state[0].size()))


if __name__ == '__main__':
    _main()
