#    Copyright (C) 2018 Anvita Gupta
#
#    This program is free software: you can redistribute it and/or  modify
#    it under the terms of the GNU Affero General Public License, version 3,
#    as published by the Free Software Foundation.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from utils.torch_utils import *

class ResBlock(nn.Module):
    def __init__(self, hidden):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)

class Generator_lang(nn.Module):
    def __init__(self, n_chars, seq_len, batch_size, hidden):
        super(Generator_lang, self).__init__()
        self.fc1 = nn.Linear(128, hidden*seq_len)
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(hidden, n_chars, 1)
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden = hidden

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.view(-1, self.hidden, self.seq_len) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(self.batch_size*self.seq_len, -1)
        output = gumbel_softmax(output, 0.5)
        return output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))

class Discriminator_lang(nn.Module):
    def __init__(self, n_chars, seq_len, batch_size, hidden):
        super(Discriminator_lang, self).__init__()
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden = hidden
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1d = nn.Conv1d(n_chars, hidden, 1)
        self.linear = nn.Linear(seq_len*hidden, 1)

    def forward(self, input):
        output = input.transpose(1, 2) # (BATCH_SIZE, len(charmap), SEQ_LEN) = (16, 21, 512)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, self.seq_len*self.hidden)
        disc_output = self.linear(output)
        return disc_output


class ResBlock_logkm(nn.Module):
    def __init__(self, hidden):
        super(ResBlock_logkm, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + output

class Discriminator_logkm(nn.Module):
    def __init__(self, n_chars, seq_len, batch_size, hidden, sub_len):
        super(Discriminator_logkm, self).__init__()
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden = hidden
        self.sub_len = sub_len
        self.block = nn.Sequential(
            ResBlock_logkm(hidden),
            ResBlock_logkm(hidden),
            ResBlock_logkm(hidden),
            ResBlock_logkm(hidden),
            ResBlock_logkm(hidden),
        )
        self.fc = nn.Linear((n_chars * seq_len) + sub_len, hidden)

        self.linear_logkm = nn.Linear(hidden, 1)

    def forward(self, input, sub):

        input = input.view(-1, self.n_chars * self.seq_len) # (BATCH_SIZE, len(charmap), SEQ_LEN)
        init_input = torch.cat([input, sub], dim=1)
        output = self.fc(init_input)
        output = self.block(output)
        es_logkm = self.linear_logkm(output)

        return es_logkm

if __name__ == "__main__":
    x = torch.randn(16, 21, 512)
    c = torch.randn(16, 1024)

    net = Discriminator_logkm(21, 512, 16, 512, 1024)

    out = net(x, c)
    print(out.shape)