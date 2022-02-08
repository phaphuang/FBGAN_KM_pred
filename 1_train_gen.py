import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

from sklearn.preprocessing import OneHotEncoder
import os, math, glob, argparse
from utils.torch_utils import *
from utils.utils import *
import matplotlib.pyplot as plt
import utils.language_helpers
plt.switch_backend('agg')
import numpy as np
from multi_models import *

class WGAN_LangGP():
    def __init__(self, batch_size=16, lr=0.0001, num_epochs=1000, seq_len=512, data_dir="./data/brenda_data_checkpoint_5.csv",
        run_name="test", hidden=512, d_steps=5, g_steps=1):

        self.hidden = hidden
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = num_epochs
        self.seq_len = seq_len
        self.d_steps = d_steps
        self.g_steps = g_steps
        self.lamda = 10

        self.checkpoint_dir = './checkpoint/' + run_name + "/"
        self.sample_dir = './sample/' + run_name + "/"
        self.load_data(data_dir)
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir): os.makedirs(self.sample_dir)
        self.use_cuda = True if torch.cuda.is_available() else False
        self.build_model()
    
    def build_model(self):
        self.G = Generator_lang(len(self.charmap), self.seq_len, self.batch_size, self.hidden)
        self.D = Discriminator_lang(len(self.charmap), self.seq_len, self.batch_size, self.hidden)
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()
        #print(self.G)
        #print(self.D)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))

    def load_data(self, datadir):
        max_examples = 1e6
        lines, self.charmap, self.inv_charmap, self.logkm_list = utils.language_helpers.load_dataset(
            max_length=self.seq_len,
            max_n_examples=max_examples,
            data_dir=datadir
        )
        self.data = lines
    
    def save_model(self, epoch):
        torch.save(self.G.state_dict(), self.checkpoint_dir + "G_weights_{}.pth".format(epoch))
        torch.save(self.D.state_dict(), self.checkpoint_dir + "D_weights_{}.pth".format(epoch))
    
    def load_model(self, directory = ''):
        '''
            Load model parameters from most recent epoch
        '''
        if len(directory) == 0:
            directory = self.checkpoint_dir
        list_G = glob.glob(directory + "G*.pth")
        list_D = glob.glob(directory + "D*.pth")
        if len(list_G) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 1 #file is not there
        G_file = max(list_G, key=os.path.getctime)
        D_file = max(list_D, key=os.path.getctime)
        epoch_found = int( (G_file.split('_')[-1]).split('.')[0])
        print("[*] Checkpoint {} found at {}!".format(epoch_found, directory))
        self.G.load_state_dict(torch.load(G_file))
        self.D.load_state_dict(torch.load(D_file))
        return epoch_found

    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1, 1)
        alpha = alpha.view(-1,1,1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.cuda() if self.use_cuda else alpha
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda() if self.use_cuda else interpolates
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda() \
                                  if self.use_cuda else torch.ones(disc_interpolates.size()),
                                  create_graph=True, retain_graph=True)[0]

        gradients = gradients.contiguous().view(self.batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        #gradient_penalty = ((gradients.norm(2, dim=1).norm(2,dim=1) - 1) ** 2).mean() * self.lamda
        return self.lamda * ((gradients_norm - 1) ** 2).mean()
    
    def disc_train_iteration(self, real_data):
        self.D_optimizer.zero_grad()

        fake_data_1 = self.sample_generator(self.batch_size)
        d_fake_pred_1 = self.D(fake_data_1)
        d_fake_err_1 = d_fake_pred_1.mean()

        fake_data_2 = self.sample_generator(self.batch_size)
        d_fake_pred_2 = self.D(fake_data_2)
        d_fake_err_2 = d_fake_pred_2.mean()

        d_fake_err = (d_fake_err_1 + d_fake_err_2) / 2

        d_real_pred = self.D(real_data)
        d_real_err = d_real_pred.mean()

        gradient_penalty = self.calc_gradient_penalty(real_data, fake_data_1)

        d_err = d_fake_err - d_real_err + gradient_penalty
        d_err.backward()
        self.D_optimizer.step()

        return d_fake_err.data, d_real_err.data, gradient_penalty.data, d_err
    
    def sample_generator(self, num_sample):
        z_input = Variable(torch.randn(num_sample, 128))
        if self.use_cuda: z_input = z_input.cuda()
        generated_data = self.G(z_input)
        return generated_data

    def gen_train_iteration(self):
        self.G.zero_grad()
        z_input = to_var(torch.randn(self.batch_size, 128))
        g_fake_data = self.G(z_input)

        z_input_2 = to_var(torch.randn(self.batch_size, 128))
        g_fake_data_2 = self.G(z_input_2)

        #print("Sample g: ", g_fake_data)
        dg_fake_pred_1 = self.D(g_fake_data)
        dg_fake_pred_2 = self.D(g_fake_data_2)
        g_err = -torch.mean((dg_fake_pred_1 + dg_fake_pred_2) / 2)

        lz = torch.mean(torch.abs(g_fake_data - g_fake_data_2)) / torch.mean(
            torch.abs(z_input - z_input_2))
        eps = 1 * 1e-5
        loss_lz = 1 / (lz + eps)

        g_err += loss_lz

        g_err.backward()
        self.G_optimizer.step()

        return g_err
    
    def train_model(self, load_dir):
        init_epoch = self.load_model(load_dir)
        n_batches = int(len(self.data)/self.batch_size)
        total_iterations = n_batches * self.n_epochs
        losses_f = open(self.checkpoint_dir + "losses.txt",'a+')
        d_fake_losses, d_real_losses, grad_penalties = [],[],[]
        G_losses, D_losses, W_dist = [],[],[]
        V_losses = []

        table = np.arange(len(self.charmap)).reshape(-1, 1)
        one_hot = OneHotEncoder()
        one_hot.fit(table)

        if init_epoch:
          counter = init_epoch + 1
        else:
          counter = 0
  
        for epoch in range(self.n_epochs):
            for idx in range(n_batches):
                _data = np.array(
                    [[self.charmap[c] for c in l] for l in self.data[idx*self.batch_size:(idx+1)*self.batch_size]],
                    dtype='int32'
                )
                data_one_hot = one_hot.transform(_data.reshape(-1, 1)).toarray().reshape(self.batch_size, -1, len(self.charmap))
                real_data = torch.Tensor(data_one_hot)
                real_data = to_var(real_data)

                d_fake_err, d_real_err, gradient_penalty, d_err = self.disc_train_iteration(real_data)

                # Append things for logging
                d_fake_np, d_real_np, gp_np = d_fake_err.cpu().numpy(), \
                        d_real_err.cpu().numpy(), gradient_penalty.cpu().numpy()
                grad_penalties.append(gp_np)
                d_real_losses.append(d_real_np)
                d_fake_losses.append(d_fake_np)
                D_losses.append(d_fake_np - d_real_np + gp_np)
                W_dist.append(d_real_np - d_fake_np)

                if counter % self.d_steps == 0:
                    g_err = self.gen_train_iteration()
                    G_losses.append((g_err.data).cpu().numpy())

                if counter % 1200 == 1199:
                    self.save_model(counter)
                    self.sample(counter, total_iterations)
                if counter % 10 == 9:
                    summary_str = 'Iteration [{}/{}] - loss_d: {}, loss_g: {}, w_dist: {}, grad_penalty: {}'\
                        .format(counter, total_iterations, (d_err.data).cpu().numpy(),
                        (g_err.data).cpu().numpy(), ((d_real_err - d_fake_err).data).cpu().numpy(), gp_np)
                    print(summary_str)
                    losses_f.write(summary_str + "\n")
                    plot_losses([G_losses, D_losses], ["gen", "disc"], self.sample_dir + "losses.png")
                    plot_losses([W_dist], ["w_dist"], self.sample_dir + "dist.png")
                    plot_losses([grad_penalties],["grad_penalties"], self.sample_dir + "grad.png")
                    plot_losses([d_fake_losses, d_real_losses],["d_fake", "d_real"], self.sample_dir + "d_loss_components.png")
                counter += 1
            np.random.shuffle(self.data)

    def sample(self, step, total_steps):
        z = to_var(torch.randn(self.batch_size, 128))
        self.G.eval()
        torch_seqs = self.G(z)
        seqs = (torch_seqs.data).cpu().numpy()
        decoded_seqs = [decode_one_seq(seq, self.inv_charmap)+"\n" for seq in seqs]
        with open(self.sample_dir + "sampled_{}.txt".format(step), 'w+') as f:
            f.writelines(decoded_seqs)

        self.G.train()

def main():
    parser = argparse.ArgumentParser(description='WGAN-GP for producing gene sequences.')
    parser.add_argument("--run_name", default= "test_brenda", help="Name for output files (checkpoint and sample dir)")
    parser.add_argument("--load_dir", default="", help="Option to load checkpoint from other model (Defaults to run name)")
    args = parser.parse_args()
    model = WGAN_LangGP(run_name=args.run_name)
    model.train_model(args.load_dir)

if __name__ == '__main__':
    main()