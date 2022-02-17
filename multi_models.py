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
from torch import LongTensor, optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from utils.torch_utils import *
import numpy as np

from utils.constants import AMINO_ACID_TO_ID
from sklearn.preprocessing import OneHotEncoder

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
        self.use_gpu = True if torch.cuda.is_available() else False
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
    
    def indexes_from_sentence(self, sentence):
        return [AMINO_ACID_TO_ID[t] for t in sentence]
    
    def pad_seq(self, seq, pad_char, length):
        padded_seq = seq + [pad_char]*(length - len(seq))
        return padded_seq
    
    def predict_model(self, input_seqs, substrates):
        PAD_token = 0
        num_pred_batches = int(len(input_seqs)/self.batch_size)
        all_preds = np.zeros((num_pred_batches*self.batch_size, 1))

        table = np.arange(len(AMINO_ACID_TO_ID)).reshape(-1, 1)
        one_hot = OneHotEncoder()
        one_hot.fit(table)

        #print("This is length: ", len(input_seqs), len(substrates))

        for idx in range(num_pred_batches):
            batch_seqs = input_seqs[idx*self.batch_size:(idx+1)*self.batch_size]
            tokenized_seqs = [self.indexes_from_sentence(s.strip()) for s in batch_seqs]
            input_lengths = [len(s) for s in tokenized_seqs]
            input_padded = [self.pad_seq(s, PAD_token, max(input_lengths)) for s in tokenized_seqs]
            input_var = Variable(torch.LongTensor(input_padded)).transpose(0,1)
            #print("Input Var shape: ", input_var.shape)
            data_one_hot = one_hot.transform(input_var.reshape(-1, 1)).toarray().reshape(self.batch_size, -1, len(AMINO_ACID_TO_ID))
            real_data = torch.Tensor(data_one_hot)
            real_data = real_data.cuda() if self.use_gpu else real_data
            
            batch_subs = substrates[idx*self.batch_size:(idx+1)*self.batch_size]

            input = real_data.view(-1, self.n_chars * self.seq_len) # (BATCH_SIZE, len(charmap), SEQ_LEN)
            init_input = torch.cat([input, batch_subs], dim=1)
            output = self.fc(init_input)
            output = self.block(output)
            es_logkm = self.linear_logkm(output)
            all_preds[idx*self.batch_size:(idx+1)*self.batch_size,:] = es_logkm.data.cpu().numpy()
        return all_preds

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, batch_size, hidden_dim):
        super(GRUClassifier, self).__init__()
        self.hidden = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, bidirectional=False, num_layers=2, dropout=0.3)
        self.linear = nn.Linear(hidden_dim, 1)
        self.batch_size = batch_size
        self.use_cuda = True if torch.cuda.is_available() else False

        self.embedding_s = nn.Embedding(2, hidden_dim)
    
    def forward(self, x, h, s):
        #print("Before embedding shape: ", x.shape, s.shape)
        x = self.embedding(x)
        s = self.embedding_s(s)
        #print("After embedding shape: ", x.shape, s.shape)

        x = torch.cat([x, s], axis=0)

        x, h = self.gru(x, h)
        x = self.linear(x[-1])
        return x, h
    
    def init_hidden(self):
        if self.use_cuda:
            return Variable(torch.randn(2, self.batch_size, self.hidden)).cuda()
        return Variable(torch.randn(2, self.batch_size, self.hidden))

if __name__ == "__main__":
    x = torch.rand(512, 16).long()
    s = torch.rand(1024, 16).long()
    
    net = GRUClassifier(21, 16, 128)
    h = net.init_hidden()
    out, h_out = net(x, h, s)
    print(out.shape)