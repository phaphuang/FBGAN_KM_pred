import argparse
from sklearn.preprocessing import OneHotEncoder
from utils.torch_utils import *
from utils.utils import *
import matplotlib.pyplot as plt
import utils.language_helpers
plt.switch_backend('agg')
import numpy as np
from multi_models import Discriminator_logkm

import os
from torch import optim
import glob

from sklearn.model_selection import train_test_split

seed = 42

class KM_Predictor():
    def __init__(self, batch_size=16, lr=0.001, num_epochs=2000, seq_len=512, data_dir="./data/new_brenda_substrate_df_data.csv",
        run_name_pred="test_pred", hidden=512, sub_len=1024):

        self.hidden = hidden
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = num_epochs
        self.seq_len = seq_len
        self.sub_len = sub_len

        self.checkpoint_pred_dir = "./checkpoint/" + run_name_pred + "/"

        self.load_data(data_dir)
        if not os.path.exists(self.checkpoint_pred_dir): os.makedirs(self.checkpoint_pred_dir)
        self.use_cuda = True if torch.cuda.is_available() else False
        self.build_model()
    
    def build_model(self):
        self.P = Discriminator_logkm(len(self.charmap), self.seq_len, self.batch_size, self.hidden, self.sub_len)
        if self.use_cuda:
            self.P.cuda()
        
        self.P_optimizer = optim.Adam(self.P.parameters(), lr=self.lr, betas=(0.5, 0.9))
        #self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
    
    def load_data(self, datadir):
        max_examples = 1e6
        self.data, self.charmap, self.inv_charmap, self.substate_ecfp, self.logkm_list = utils.language_helpers.load_dataset_ecfp(
            max_length=self.seq_len,
            max_n_examples=max_examples,
            data_dir=datadir
        )
    
    def save_model(self, epoch):
        torch.save(self.P.state_dict(), self.checkpoint_pred_dir + "P_weights_{}.pth".format(epoch))
    
    def load_model(self, director_valid = ''):
        '''
            Load model parameters from most recent epoch
        '''
        if len(director_valid) == 0:
            directory_valid = self.checkpoint_pred_dir
  
        list_P = glob.glob(directory_valid + "P*.pth")
        if len(list_P) == 0:
            print("[*] Predictor Checkpoint not found! Starting from scratch.")
            return 1
        P_file = max(list_P, key=os.path.getctime)
        epoch_pred_found = int( (P_file.split('_')[-1]).split('.')[0])
        print("[*] Predictor Checkpoint {} found at {}!".format(epoch_pred_found, director_valid))
        self.P.load_state_dict(torch.load(P_file))

        return epoch_pred_found
    
    def pred_train_iteration(self, real_data, substrate_data, logkm_data):
        self.P_optimizer.zero_grad()

        logkm_pred = self.P(real_data, substrate_data)
        logkm_err = self.criterion(logkm_data, logkm_pred.squeeze())

        logkm_err.backward()
        self.P_optimizer.step()

        return logkm_err.data
    
    def train_model(self, load_dir_valid):
        init_epoch =self.load_model(load_dir_valid)
        losses_f = open(self.checkpoint_pred_dir + "pred_losses.txt",'a+')
        losses_f_eval = open(self.checkpoint_pred_dir + "eval_pred_losses.txt",'a+')
        P_losses = []

        table = np.arange(len(self.charmap)).reshape(-1,1)
        one_hot = OneHotEncoder()
        one_hot.fit(table)

        if init_epoch:
            counter = init_epoch + 1
        else:
            counter = 0
        
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.logkm_list, test_size=0.2, random_state=seed)
        n_train_batches = int(len(X_train)/self.batch_size)
        total_train_iteration = n_train_batches * self.n_epochs

        n_test_batches = int(len(X_test)/self.batch_size)
        total_test_iteration = n_test_batches * self.n_epochs
        
        for epoch in range(self.n_epochs):
            for idx in range(n_train_batches):
                _data = np.array(
                    [[self.charmap[c] for c in l] for l in X_train[idx*self.batch_size:(idx+1)*self.batch_size]],
                    dtype='int32'
                )
                data_one_hot = one_hot.transform(_data.reshape(-1, 1)).toarray().reshape(self.batch_size, -1, len(self.charmap))
                real_data = torch.Tensor(data_one_hot)
                real_data = to_var(real_data)

                logkm_data = torch.Tensor(np.array(y_train[idx*self.batch_size:(idx+1)*self.batch_size]))
                logkm_data = to_var(logkm_data)

                substrate_data = torch.Tensor(np.array(
                    [[c for c in l] for l in self.substate_ecfp[idx*self.batch_size:(idx+1)*self.batch_size]],
                    dtype="int32"
                ))
                substrate_data = to_var(substrate_data)

                logkm_err = self.pred_train_iteration(real_data, substrate_data, logkm_data)

                P_losses.append(logkm_err.item())

                if counter % 5000 == 0:
                    self.save_model(counter)

                counter += 1

            summary_str = 'Train Epoch [{}/{}] Iteration [{}/{}] - loss_km: {}'\
                .format(epoch, self.n_epochs, counter, total_train_iteration, np.mean(P_losses))
            print(summary_str)
            losses_f.write(summary_str + "\n")
            plot_losses([P_losses], ["logkm"], self.checkpoint_pred_dir + "logkm.png")
            
            self.P.eval()
            P_losses_eval = []
            for idx in range(n_test_batches):
                _data = np.array(
                    [[self.charmap[c] for c in l] for l in X_test[idx*self.batch_size:(idx+1)*self.batch_size]],
                    dtype='int32'
                )
                data_one_hot = one_hot.transform(_data.reshape(-1, 1)).toarray().reshape(self.batch_size, -1, len(self.charmap))
                real_data = torch.Tensor(data_one_hot)
                real_data = to_var(real_data)

                logkm_data = torch.Tensor(np.array(y_test[idx*self.batch_size:(idx+1)*self.batch_size]))
                logkm_data = to_var(logkm_data)

                substrate_data = torch.Tensor(np.array(
                    [[c for c in l] for l in self.substate_ecfp[idx*self.batch_size:(idx+1)*self.batch_size]],
                    dtype="int32"
                ))
                substrate_data = to_var(substrate_data)

                logkm_pred = self.P(real_data, substrate_data)
                logkm_err = self.criterion(logkm_data, logkm_pred.squeeze())

                P_losses_eval.append(logkm_err.item())


            summary_str = 'Test Epoch [{}/{}] - loss_km: {}'\
                        .format(epoch, self.n_epochs, np.mean(P_losses_eval))
            print(summary_str)
            losses_f_eval.write(summary_str + "\n")

def main():
    parser = argparse.ArgumentParser(description='Predictor for estimating KM values')
    parser.add_argument("--run_name_pred", default="test_pred", help="Name for output file (predictor checkpoint)")
    parser.add_argument("--load_dir_valid", default="", help="Option to load validator checkpoint from other model")
    args = parser.parse_args()
    model = KM_Predictor(run_name_pred=args.run_name_pred)
    model.train_model(args.load_dir_valid)

if __name__ == '__main__':
    main()
