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
import random

class WGAN_LangGP():
    def __init__(self, batch_size=16, lr=0.0001, num_epochs=1000, seq_len=512, data_dir="./data/new_brenda_substrate_df_data.csv",
        run_name="test", run_name_pred="test_pred", hidden=512, d_steps=10, g_steps=1, sub_len=1024):

        self.preds_cutoff = 0
        self.hidden = hidden
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = num_epochs
        self.seq_len = seq_len
        self.d_steps = d_steps
        self.g_steps = g_steps
        self.lamda = 10

        self.sub_len = sub_len

        self.checkpoint_dir = './checkpoint/' + run_name + "/"
        self.sample_dir = './sample/' + run_name + "/"

        self.checkpoint_pred_dir = "./checkpoint/" + run_name_pred + "/"

        self.load_data(data_dir)
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.checkpoint_pred_dir): os.makedirs(self.checkpoint_pred_dir)
        if not os.path.exists(self.sample_dir): os.makedirs(self.sample_dir)
        self.use_cuda = True if torch.cuda.is_available() else False
        self.build_model()
    
    def build_model(self):
        self.G = Generator_lang(len(self.charmap), self.seq_len, self.batch_size, self.hidden)
        self.D = Discriminator_lang(len(self.charmap), self.seq_len, self.batch_size, self.hidden)

        self.P = Discriminator_logkm(len(self.charmap), self.seq_len, self.batch_size, self.hidden, self.sub_len)
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()
            self.P.cuda()
        #print(self.G)
        #print(self.D)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))

        self.P_optimizer = optim.Adam(self.P.parameters(), lr=self.lr, betas=(0.5, 0.9))

        self.criterion = nn.MSELoss()

    def load_data(self, datadir):
        max_examples = 1e6
        self.data, self.charmap, self.inv_charmap, self.substate_ecfp, self.logkm_list = utils.language_helpers.load_dataset_ecfp(
            max_length=self.seq_len,
            max_n_examples=max_examples,
            data_dir=datadir
        )

        self.labels = np.zeros(len(self.data))
    
    def remove_old_indices(self, numToAdd):
        toRemove = np.argsort(self.labels)[:numToAdd]
        self.data = [d for i,d in enumerate(self.data) if i not in toRemove]
        self.labels = np.delete(self.labels, toRemove)
    
    def save_model(self, epoch):
        torch.save(self.G.state_dict(), self.checkpoint_dir + "G_weights_{}.pth".format(epoch))
        torch.save(self.D.state_dict(), self.checkpoint_dir + "D_weights_{}.pth".format(epoch))
    
    def load_model(self, directory = '', director_pred = ''):
        '''
            Load model parameters from most recent epoch
        '''
        if len(directory) == 0:
            directory = self.checkpoint_dir
        list_G = glob.glob(directory + "G*.pth")
        list_D = glob.glob(directory + "D*.pth")
        
        if len(list_G) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 1
        
        G_file = max(list_G, key=os.path.getctime)
        D_file = max(list_D, key=os.path.getctime)
        epoch_gen_found = int( (G_file.split('_')[-1]).split('.')[0])
        print("[*] Checkpoint {} found at {}!".format(epoch_gen_found, directory))
        self.G.load_state_dict(torch.load(G_file))
        self.D.load_state_dict(torch.load(D_file))

        ##### Loading validator #####
        if len(director_pred) == 0:
            directory_pred = self.checkpoint_pred_dir
  
        list_P = glob.glob(directory_pred + "P*.pth")
        if len(list_P) == 0:
            print("[*] Predictor Checkpoint not found! Starting from scratch.")
        else:
            P_file = max(list_P, key=os.path.getctime)
            self.P.load_state_dict(torch.load(P_file))
            epoch_pred_found = int( (P_file.split('_')[-1]).split('.')[0])
            print("[*] Predictor Checkpoint {} found at {}!".format(epoch_pred_found, directory_pred))

        return epoch_gen_found

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

        fake_data = self.sample_generator(self.batch_size)
        d_fake_pred = self.D(fake_data)
        d_fake_err = d_fake_pred.mean()
        d_real_pred = self.D(real_data)
        d_real_err = d_real_pred.mean()

        gradient_penalty = self.calc_gradient_penalty(real_data, fake_data)

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
        dg_fake_pred = self.D(g_fake_data)
        g_err = -torch.mean(dg_fake_pred)
        g_err.backward()
        self.G_optimizer.step()
        return g_err
    
    def train_model(self, load_dir, load_dir_pred):
        init_epoch = self.load_model(load_dir, load_dir_pred)
        n_batches = int(len(self.data)/self.batch_size)
        total_iterations = n_batches * self.n_epochs
        losses_f = open(self.checkpoint_dir + "losses.txt",'a+')
        d_fake_losses, d_real_losses, grad_penalties = [],[],[]
        G_losses, D_losses, W_dist = [],[],[]

        table = np.arange(len(self.charmap)).reshape(-1, 1)
        one_hot = OneHotEncoder()
        one_hot.fit(table)

        if init_epoch:
          counter = init_epoch + 1
        else:
          counter = 0
        
        num_batches_sample = 15
        n_batches = int(len(self.data)/self.batch_size)
  
        for epoch in range(1, self.n_epochs + 1):

            rand_sub = []

            if epoch % 2 == 0: self.save_model(epoch)
            sampled_seqs = self.sample(num_batches_sample, epoch)
            sampled_sub = random.sample(self.substate_ecfp, num_batches_sample*self.batch_size)
            sampled_sub = torch.Tensor(np.array(
                    [[c for c in l] for l in sampled_sub],
                    dtype="int32"
            ))
            sampled_sub = to_var(sampled_sub)
            preds = self.P.predict_model(sampled_seqs, sampled_sub)

            with open(self.sample_dir + "sampled_{}_preds.txt".format(epoch), 'w+') as f:
                f.writelines([s + '\t' + str(preds[j][0]) + '\n' for j, s in enumerate(sampled_seqs)])
            good_indices = (preds > self.preds_cutoff).nonzero()[0]
            pos_seqs = [list(sampled_seqs[i]) for i in good_indices]
            print("Adding {} positive sequences".format(len(pos_seqs)))
            with open(self.checkpoint_dir + "positives.txt",'a+') as f:
                f.write("Epoch: {} \t Pos: {}\n".format(epoch, len(pos_seqs)/float(len(sampled_seqs))))
            self.remove_old_indices(len(pos_seqs))
            self.data += pos_seqs
            self.labels = np.concatenate([self.labels, np.repeat(epoch, len(pos_seqs))] )
            perm = np.random.permutation(len(self.data))
            self.data = [self.data[i] for i in perm]
            self.labels = self.labels[perm]

            for idx in range(n_batches):
                _data = np.array(
                    [[self.charmap[c] for c in l] for l in self.data[idx*self.batch_size:(idx+1)*self.batch_size]],
                    dtype='int32'
                )
                data_one_hot = one_hot.transform(_data.reshape(-1, 1)).toarray().reshape(self.batch_size, -1, len(self.charmap))
                real_data = torch.Tensor(data_one_hot)
                real_data = to_var(real_data)

                logkm_data = torch.Tensor(np.array(self.logkm_list[idx*self.batch_size:(idx+1)*self.batch_size]))
                logkm_data = to_var(logkm_data)

                substrate_data = torch.Tensor(np.array(
                    [[c for c in l] for l in self.substate_ecfp[idx*self.batch_size:(idx+1)*self.batch_size]],
                    dtype="int32"
                ))
                substrate_data = to_var(substrate_data)

                #### Training Discriminator ####
                for p in self.D.parameters():  # reset requires_grad
                    p.requires_grad = True
                for _ in range(self.d_steps):
                    d_fake_err, d_real_err, gradient_penalty, d_err = self.disc_train_iteration(real_data)

                # Append things for logging
                d_fake_np, d_real_np, gp_np = d_fake_err.cpu().numpy(), \
                        d_real_err.cpu().numpy(), gradient_penalty.cpu().numpy()
                grad_penalties.append(gp_np)
                d_real_losses.append(d_real_np)
                d_fake_losses.append(d_fake_np)
                D_losses.append(d_fake_np - d_real_np + gp_np)
                W_dist.append(d_real_np - d_fake_np)

                #### Train Generator ####
                for p in self.D.parameters():
                    p.requires_grad = False
                g_err = self.gen_train_iteration()
                G_losses.append((g_err.data).cpu().numpy())

                if counter % 100 == 99:
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

    def sample(self, num_batches_sample, epoch):
        decoded_seqs = []
        for i in range(num_batches_sample):
            z = to_var(torch.randn(self.batch_size, 128))
            self.G.eval()
            torch_seqs = self.G(z)
            seqs = (torch_seqs.data).cpu().numpy()
            decoded_seqs += [decode_one_seq(seq, self.inv_charmap) for seq in seqs]
        self.G.train()
        return decoded_seqs
    

def main():
    parser = argparse.ArgumentParser(description='WGAN-GP for producing gene sequences.')
    parser.add_argument("--run_name", default= "test_brenda", help="Name for output files (checkpoint and sample dir)")
    parser.add_argument("--run_name_pred", default="test_pred", help="Name for output file (predictor checkpoint)")
    parser.add_argument("--load_dir", default="", help="Option to load checkpoint from other model (Defaults to run name)")
    parser.add_argument("--load_dir_pred", default="", help="Option to load predictor checkpoint from other model (Defaults to run name)")
    args = parser.parse_args()
    model = WGAN_LangGP(run_name=args.run_name, run_name_pred=args.run_name_pred)
    model.train_model(args.load_dir, args.load_dir_pred)

if __name__ == '__main__':
    main()