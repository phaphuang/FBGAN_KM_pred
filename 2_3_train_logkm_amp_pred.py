import argparse
from sklearn.preprocessing import OneHotEncoder
from utils.torch_utils import *
from utils.utils import *
import matplotlib.pyplot as plt
import utils.language_helpers
plt.switch_backend('agg')
import numpy as np
from multi_models import GRUClassifier

import os
from torch import optim
import glob

from sklearn.model_selection import train_test_split

seed = 42

def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))

class KM_Predictor():
    def __init__(self, batch_size=16, lr=0.001, num_epochs=2000, seq_len=512, data_dir="./data/new_brenda_substrate_df_data.csv",
        run_name_pred="test_pred", hidden=128, sub_len=1024):

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
        self.rnn = GRUClassifier(len(self.charmap), self.batch_size, hidden_dim=self.hidden)
        if self.use_cuda:
            self.rnn.cuda()
        
        self.rnn_optimizer = optim.Adam(self.rnn.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def load_data(self, datadir):
        max_examples = 1e6
        self.data, self.charmap, self.inv_charmap, self.substate_ecfp, self.logkm_list = utils.language_helpers.load_dataset_ecfp(
            max_length=self.seq_len,
            max_n_examples=max_examples,
            data_dir=datadir
        )
    
    def save_model(self, epoch):
        torch.save(self.rnn.state_dict(), self.checkpoint_pred_dir + "rnn_weights_{}.pth".format(epoch))
    
    def load_model(self, director_valid = ''):
        '''
            Load model parameters from most recent epoch
        '''
        if len(director_valid) == 0:
            directory_valid = self.checkpoint_pred_dir
  
        list_rnn = glob.glob(directory_valid + "rnn*.pth")
        if len(list_rnn) == 0:
            print("[*] Predictor Checkpoint not found! Starting from scratch.")
            return 1
        rnn_file = max(list_rnn, key=os.path.getctime)
        epoch_pred_found = int( (rnn_file.split('_')[-1]).split('.')[0])
        print("[*] Predictor Checkpoint {} found at {}!".format(epoch_pred_found, director_valid))
        self.rnn.load_state_dict(torch.load(list_rnn))

        return epoch_pred_found
    
    def pred_train_iteration(self, real_data, h, substrate_data, logkm_data):
        self.rnn_optimizer.zero_grad()

        logkm_pred, h = self.rnn(real_data, h, substrate_data)
        logkm_err = self.criterion(logkm_data, logkm_pred.squeeze())

        logkm_err.backward()
        self.rnn_optimizer.step()

        return logkm_err.data, h
    
    def train_model(self, load_dir_valid):
        init_epoch =self.load_model(load_dir_valid)
        losses_f = open(self.checkpoint_pred_dir + "pred_losses.txt",'a+')
        losses_f_eval = open(self.checkpoint_pred_dir + "eval_pred_losses.txt",'a+')
        P_losses = []

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
            h = self.rnn.init_hidden()
            for idx in range(n_train_batches):
                _data = np.array(
                    [[self.charmap[c] for c in l] for l in X_train[idx*self.batch_size:(idx+1)*self.batch_size]],
                    dtype='int32'
                )
                _data = _data.transpose(1,0)
                real_data = torch.LongTensor(_data)
                real_data = to_var(real_data)

                logkm_data = torch.Tensor(np.array(y_train[idx*self.batch_size:(idx+1)*self.batch_size]))
                logkm_data = to_var(logkm_data)

                _sdata = np.array(
                    [[c for c in l] for l in self.substate_ecfp[idx*self.batch_size:(idx+1)*self.batch_size]],
                    dtype="int32"
                )
                _sdata = _sdata.transpose(1,0)
                substrate_data = torch.LongTensor(_sdata)
                substrate_data = to_var(substrate_data)

                h.detach_()
                logkm_err, h = self.pred_train_iteration(real_data, h, substrate_data, logkm_data)

                P_losses.append(logkm_err.item())

                if counter % 5000 == 0:
                    self.save_model(counter)

                counter += 1

            summary_str = 'Train Epoch [{}/{}] Iteration [{}/{}] - MSE loss_km: {}'\
                .format(epoch, self.n_epochs, counter, total_train_iteration, np.mean(P_losses))
            print(summary_str)
            losses_f.write(summary_str + "\n")
            plot_losses([P_losses], ["logkm"], self.checkpoint_pred_dir + "logkm.png")
            
            self.rnn.eval()
            P_losses_eval = []
            P_losses_eval_rmse = []
            hid = self.rnn.init_hidden()
            for idx in range(n_test_batches):
                _data = np.array(
                    [[self.charmap[c] for c in l] for l in X_test[idx*self.batch_size:(idx+1)*self.batch_size]],
                    dtype='int32'
                )
                _data = _data.transpose(1,0)
                real_data = torch.LongTensor(_data)
                real_data = to_var(real_data)

                logkm_data = torch.Tensor(np.array(y_test[idx*self.batch_size:(idx+1)*self.batch_size]))
                logkm_data = to_var(logkm_data)

                _sdata = np.array(
                    [[c for c in l] for l in self.substate_ecfp[idx*self.batch_size:(idx+1)*self.batch_size]],
                    dtype="int32"
                )
                _sdata = _sdata.transpose(1,0)
                substrate_data = torch.LongTensor(_sdata)
                substrate_data = to_var(substrate_data)

                logkm_pred, hid = self.rnn(real_data, hid, substrate_data)
                logkm_err = self.criterion(logkm_data, logkm_pred.squeeze())

                P_losses_eval.append(logkm_err.item())

                rmse_loss = RMSELoss(logkm_pred.squeeze(), logkm_data)
                P_losses_eval_rmse.append(rmse_loss.item())

            summary_str = 'Test Epoch [{}/{}] - MSE loss_km: {} RMSE loss_km: {}'\
                        .format(epoch, self.n_epochs, np.mean(P_losses_eval), np.mean(P_losses_eval_rmse))
            print(summary_str)
            losses_f_eval.write(summary_str + "\n")

def main():
    parser = argparse.ArgumentParser(description='Predictor for estimating KM values')
    parser.add_argument("--run_name_pred", default="test_pred_amp", help="Name for output file (predictor checkpoint)")
    parser.add_argument("--load_dir_valid", default="", help="Option to load validator checkpoint from other model")
    args = parser.parse_args()
    model = KM_Predictor(run_name_pred=args.run_name_pred)
    model.train_model(args.load_dir_valid)

if __name__ == '__main__':
    main()