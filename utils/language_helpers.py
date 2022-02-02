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

import collections
import numpy as np
import re
import torch.nn.functional as F
import torch
from .constants import ID_TO_AMINO_ACID, AMINO_ACID_TO_ID, NON_STANDARD_AMINO_ACIDS
import os
import pandas as pd

def tokenize_string(sample):
    return tuple(sample.lower().split(' '))

def get_seq_str_in_fasta(sequence, id, escape=False, strip_zeros=False):

        sequence = "".join([s for s in sequence])

        if strip_zeros:
            sequence = sequence.replace("0", "")
        header = ""
        if escape:
            prefix = "\>"
        else:
            prefix = ">"

        return "{}{} {} {}{}".format(prefix, id, header, os.linesep, sequence)

def sequences_str_to_fasta(sequences, id_to_enzyme_class, escape=True, strip_zeros=False):
    return os.linesep.join([get_seq_str_in_fasta(seq, idx, escape=escape, strip_zeros=strip_zeros) for idx, seq in enumerate(sequences)])

def load_dataset(max_length, max_n_examples, tokenize=False, max_vocab_size=2048, data_dir=''):
    '''Adapted from https://github.com/igul222/improved_wgan_training/blob/master/language_helpers.py'''
    print ("loading dataset...")

    temp_df = pd.read_csv(data_dir)
    
    if len(temp_df) > max_n_examples:
        temp_df = temp_df[:max_n_examples]
    temp_seq = list(temp_df["Sequence"])

    temp_logkm = list(temp_df["log10_KM"])

    lines = []
    seqs = []

    for line, logkm in zip(temp_seq, temp_logkm):
        line = line.split(" ")[0].strip()
        if ~any(ext in line for ext in NON_STANDARD_AMINO_ACIDS):
            if tokenize:
                line = tokenize_string(line)
            else:
                line = tuple(line)

            if len(line) > max_length:
                line = line[:max_length]
            
            lines.append([line + ( ("0",)*(max_length-len(line)) ), logkm] )
            seqs.append(line + ( ("0",)*(max_length-len(line)) ))
    
    #### Export to fasta file
    fasta = sequences_str_to_fasta(seqs, id_to_enzyme_class=None, escape=False, strip_zeros=True)
    query_path = os.path.join("./", "exported_brenda_multi_class.fasta")
    with open(query_path, "w+") as f:
        f.write(fasta)

    #### Define character mapping
    np.random.shuffle(lines)

    charmap = AMINO_ACID_TO_ID
    inv_charmap = ['0','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

    filtered_lines = []
    logkm_lists = []
    for line, logkm in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                # convert all characters to '0' if not exist in inv_charmap
                filtered_line.append('0')
        filtered_lines.append(tuple(filtered_line))
        logkm_lists.append(logkm)

    for i in range(1):
        #print(lines[i])
        print(filtered_lines[i])

    print("loaded {} lines in dataset".format(len(lines)))
    # print(charmap, inv_charmap) # {'P': 0, 'A': 1, 'T': 2, 'G': 3, 'C': 4} ['P', 'A', 'T', 'G', 'C']
    return filtered_lines, charmap, inv_charmap, logkm_lists

def load_dataset_ecfp(max_length, max_n_examples, tokenize=False, max_vocab_size=2048, data_dir=''):
    '''Adapted from https://github.com/igul222/improved_wgan_training/blob/master/language_helpers.py'''
    print ("loading dataset...")

    temp_df = pd.read_csv(data_dir)
    temp_df = temp_df.dropna(subset=["ECFP"])
    
    if len(temp_df) > max_n_examples:
        temp_df = temp_df[:max_n_examples]
    
    temp_seq = list(temp_df["Sequence"])
    temp_ectp = list(temp_df["ECFP"])
    temp_logkm = list(temp_df["log10_KM"])

    lines = []
    seqs = []

    for line, ecfp_line, logkm in zip(temp_seq, temp_ectp, temp_logkm):
        line = line.split(" ")[0].strip()
        if ~any(ext in line for ext in NON_STANDARD_AMINO_ACIDS):
            if tokenize:
                line = tokenize_string(line)
            else:
                line = tuple(line)

            if len(line) > max_length:
                line = line[:max_length]
            
            lines.append([line + ( ("0",)*(max_length-len(line)) ), ecfp_line, logkm] )
            seqs.append(line + ( ("0",)*(max_length-len(line)) ))
    
    #### Export to fasta file
    fasta = sequences_str_to_fasta(seqs, id_to_enzyme_class=None, escape=False, strip_zeros=True)
    query_path = os.path.join("./", "exported_brenda_multi_class.fasta")
    with open(query_path, "w+") as f:
        f.write(fasta)

    #### Define character mapping
    np.random.shuffle(lines)

    charmap = AMINO_ACID_TO_ID
    inv_charmap = ['0','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

    filtered_lines = []
    logkm_lists = []
    ecfp_lists = []
    for line, ecfp_line, logkm in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                # convert all characters to '0' if not exist in inv_charmap
                filtered_line.append('0')
        filtered_lines.append(tuple(filtered_line))
        ecfp_lists.append(ecfp_line)
        logkm_lists.append(logkm)

    for i in range(1):
        #print(lines[i])
        print(filtered_lines[i])

    print("loaded {} lines in dataset".format(len(lines)))
    # print(charmap, inv_charmap) # {'P': 0, 'A': 1, 'T': 2, 'G': 3, 'C': 4} ['P', 'A', 'T', 'G', 'C']
    return filtered_lines, charmap, inv_charmap, ecfp_lists, logkm_lists

def convert_real_to_one_hot(real_x):
    real_to_display = F.one_hot(real_x, 11)
    #real_to_display = np.transpose(real_to_display, [0, 2, 1])
    return real_to_display


if __name__ == '__main__':
    x = [0, 1, 2, 4, 4, 2]
    x = torch.tensor(x)
    out = convert_real_to_one_hot(x)
    print(out)
    print(out.shape)