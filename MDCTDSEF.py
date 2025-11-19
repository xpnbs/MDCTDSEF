import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import Counter
from itertools import chain
import json
from typing import List
import sentencepiece as spm
from yacs.config import CfgNode as CN
import torch.nn.functional as F
import math
import random

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr, n_warmup_steps, final_step, cfg):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = lr
        self.final_step = final_step
        self.lr = lr
        self.lr_factor = cfg.TRAIN.LR_FACTOR
        self.lr_step = cfg.TRAIN.LR_STEP

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def get_lr(self):
        "Zero out the gradients by the inner optimizer"
        return self.lr

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1

        # cosine
        # if self.n_current_steps < self.n_warmup_steps:
        #     # linear warmup
        #     lr_mult = float(self.n_current_steps) / float(max(1, self.n_warmup_steps))
        # else:
        #     # cosine learning rate decay
        #     progress = float(self.n_current_steps - self.n_warmup_steps) / float(max(1, self.final_step - self.n_warmup_steps))
        #     lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        # self.lr = self.init_lr * lr_mult

        # linear decay
        if self.n_current_steps <= self.n_warmup_steps:
            # linear warmup
            lr_mult = float(self.n_current_steps) / float(max(1, self.n_warmup_steps))
            self.lr = self.init_lr * lr_mult
        else:
            if self.n_current_steps % self.lr_step == 0:
                self.lr = self.lr * self.lr_factor

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.lr


def mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


class Vocab(object):

    def __init__(self, word2id=None):
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0
            self.word2id['<sep>'] = 1
            self.word2id['<unk>'] = 2
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        vocab_entry = Vocab()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry

    @staticmethod
    def from_subword_list(subword_list):
        vocab = Vocab()
        for subword in subword_list:
            vocab.add(subword)
        return vocab

    def save(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(dict(word2id=self.word2id), f, indent=2)

    @staticmethod
    def load(file_path):
        entry = json.load(open(file_path, 'r'))
        word2id = entry['word2id']

        return Vocab(word2id)


def pad_sents(sents, pad_token):
    sents_padded = []

    max_sent_count = max([len(sent) for sent in sents])
    for sent in sents:
        if len(sent) < max_sent_count:
            leftovers = [pad_token] * (max_sent_count - len(sent))
            sent.extend(leftovers)
        sents_padded.append(sent)
    return sents_padded


def get_vocab_list(file_path, source, vocab_size):
    spm.SentencePieceTrainer.train(input=file_path, model_prefix=source, vocab_size=vocab_size)  # train the spm model
    sp = spm.SentencePieceProcessor()  # create an instance; this saves .model and .vocab files
    sp.load('{}.model'.format(source))  # loads tgt.model or src.model
    sp_list = [sp.id_to_piece(piece_id) for piece_id in range(sp.get_piece_size())]  # this is the list of subwords
    return sp_list


class BasicDataset_reg(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.drug_info = pd.read_csv(cfg.DATASET.DRUG_INFO_PATH, usecols=cfg.DATASET.DRUG_INFO_COL.split(' ')).to_numpy()
        self.drug_se = pd.read_csv(cfg.DATASET.DRUG_SE_PATH, usecols=cfg.DATASET.DRUG_SE_COL.split(' ')).to_numpy().astype(str)

        self.smiles_vocab = Vocab.load(cfg.DATASET.SMILES_VOCAB)

        self.d_combined_sim = np.load(cfg.DATASET.D_COMBINE_SIM)
        self.d_morgan_r1_sim = np.load(cfg.DATASET.D_MORGAN_R1_SIM)
        self.d_morgan_r2_sim = np.load(cfg.DATASET.D_MORGAN_R2_SIM)
        self.d_protein_onehot_sim = np.load(cfg.DATASET.D_PROTEIN_ONEHOT_SIM)
        self.d_protein_weight_sim = np.load(cfg.DATASET.D_PROTEIN_WEIGHT_SIM)

        self.se_dag_sim = np.load(cfg.DATASET.SE_DAG_SIM)
        self.se_dag_p_sim = np.load(cfg.DATASET.SE_DAG_P_SIM)

        self.se_sim_lookup = pd.read_csv(cfg.DATASET.SE_SIM_LOOKUP, header=None).iloc[:, 0].to_numpy()
        self.cids_list = self.drug_info[:,0]
        self.se_d_sim, self.se_d_freq_sim, self.d_se_sim, self.d_se_freq_sim = self._generate_freq_sim()

    def __len__(self):
        return self.drug_se.shape[0]

    def __getitem__(self, idx):
        drug_se_rela = self.drug_se[idx, :]
        drug_se_freq = drug_se_rela[2].astype(np.float)
        smiles = self._get_drug_info(drug_se_rela[0])

        for_mask = []
        smiles_index_list = []
        for i in smiles:
            smiles_index_list.append(self.smiles_vocab[i])
        smiles_index_list = smiles_index_list[:self.cfg.DATASET.SMILES_LEN]
        for_mask.extend([1 for _ in range(len(smiles_index_list))])
        padding = [self.smiles_vocab['<pad>'] for _ in range(self.cfg.DATASET.SMILES_LEN - len(smiles_index_list))]
        for_mask.extend([0 for _ in range(self.cfg.DATASET.SMILES_LEN - len(smiles_index_list))])
        smiles_index_list = smiles_index_list + padding

        drug_sim_array, se_sim_array = self._get_similarity(drug_se_rela[0], drug_se_rela[1])

        output = {'drug_sim':torch.tensor(drug_sim_array, dtype=torch.float),
                  'se_sim':torch.tensor(se_sim_array, dtype=torch.float),
                  'smiles_index_tensor':torch.tensor(smiles_index_list, dtype=torch.long),
                  'for_mask':torch.tensor(for_mask, dtype=torch.long),
                  'pos_num':torch.arange(self.cfg.DATASET.SMILES_LEN),
                  'true_freq':torch.tensor(drug_se_freq, dtype=torch.float),
                  'freq_class_index': torch.tensor(drug_se_freq - 1).long()}
        return output

    def _get_drug_info(self, cids):
        drug_info_idx = np.where(self.drug_info[:, 0] == cids)[0][0]
        smiles = self.drug_info[drug_info_idx, 1]

        return smiles

    def _generate_freq_sim(self):
        se_d_relation = np.load(self.cfg.DATASET.SE_D_SIM)
        se_d_freq_relation = np.load(self.cfg.DATASET.SE_D_FREQ_SIM)
        se_d_sim = cosine_similarity(se_d_relation)
        se_d_freq_sim = cosine_similarity(se_d_freq_relation)
        d_se_sim = cosine_similarity(se_d_relation.T)
        d_se_freq_sim = cosine_similarity(se_d_freq_relation.T)
        return se_d_sim, se_d_freq_sim, d_se_sim, d_se_freq_sim

    def _get_similarity(self, cids, se_name):
        cids_idx = np.where(self.cids_list == cids)[0][0]
        se_idx = np.where(self.se_sim_lookup == se_name)[0][0]

        d_se_sim = self.d_se_sim[cids_idx, :]
        d_se_freq_sim = self.d_se_freq_sim[cids_idx, :]
        d_combined_sim = self.d_combined_sim[cids_idx, :]
        d_morgan_r1_sim = self.d_morgan_r1_sim[cids_idx, :]
        d_morgan_r2_sim = self.d_morgan_r2_sim[cids_idx, :]
        d_protein_onehot_sim = self.d_protein_onehot_sim[cids_idx, :]
        d_protein_weight_sim = self.d_protein_weight_sim[cids_idx, :]

        drug_sim_array = np.stack((d_se_sim, d_se_freq_sim, d_combined_sim, d_morgan_r1_sim,
                                   d_morgan_r2_sim, d_protein_onehot_sim, d_protein_weight_sim), axis=0)

        assert drug_sim_array.shape == (7, 757), f'the wrong shape is {drug_sim_array.shape}'

        # side effect similarity
        se_d_sim = self.se_d_sim[se_idx, :]
        se_d_freq_sim = self.se_d_freq_sim[se_idx, :]
        se_dag_sim = self.se_dag_sim[se_idx, :]
        se_dag_p_sim = self.se_dag_p_sim[se_idx, :]

        se_sim_array = np.stack((se_d_sim, se_d_freq_sim, se_dag_sim, se_dag_p_sim), axis=0)
        assert se_sim_array.shape == (4, 994), f'the wrong shape is {se_sim_array.shape}'

        return drug_sim_array, se_sim_array


def set_config_reg():
    # config
    cfg = CN()

    cfg.EXP_NAME = 'regression'
    cfg.GPUS = "'0, 1'"
    cfg.WORKERS = 16
    cfg.PIN_MEMORY = True

    cfg.MODEL = CN()
    cfg.MODEL.HIDDEN = 128
    cfg.MODEL.LAYERS = 3
    cfg.MODEL.ATTEN_HEAD = 4
    cfg.MODEL.DROPOUT = 0.1
    cfg.MODEL.INITIAL_FEATURE_LEN = 256
    cfg.MODEL.FINAL_FEATURE_LEN = 256

    cfg.DATASET = CN()
    cfg.DATASET.VOCAB_ROOT = './data'
    cfg.DATASET.VOCAB_SMILES = 48
    cfg.DATASET.DRUG_INFO_PATH = './data/drug_info.csv'
    cfg.DATASET.DRUG_SE_PATH = './data/drug_se_freq.csv'
    cfg.DATASET.ON_MEMORY = True
    cfg.DATASET.DRUG_INFO_COL = 'CIDS SMILES'
    cfg.DATASET.DRUG_SE_COL = 'CIDS SE_NAME SE_FREQ SE_ONTOLOGY'
    cfg.DATASET.SMILES_LEN = 100
    cfg.DATASET.SIMILARITY_ROOT_PATH = './data'
    cfg.DATASET.DRUG_LEN = 757
    cfg.DATASET.SE_LEN = 994

    cfg.TRAIN = CN()
    cfg.TRAIN.EPOCHS = 150
    cfg.TRAIN.BATCH_SIZE = 400
    cfg.TRAIN.LR = 0.0001
    cfg.TRAIN.LR_FACTOR = 0.97
    cfg.TRAIN.LR_STEP = 85
    cfg.TRAIN.WEIGHT_DECAY = 0.01
    cfg.TRAIN.WARMUP_STEPS = 10
    cfg.TRAIN.FOLD = 10

    cfg.defrost()

    # drug vocab
    cfg.DATASET.SMILES_VOCAB = cfg.DATASET.VOCAB_ROOT + '/smiles_subword.json'

    # similarity
    cfg.DATASET.D_COMBINE_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_combined_score_similarity.npy'
    cfg.DATASET.D_MORGAN_R1_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_morgan_radius1_similarity.npy'
    cfg.DATASET.D_MORGAN_R2_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_morgan_radius2_similarity.npy'
    cfg.DATASET.D_PROTEIN_ONEHOT_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_protein_onehot_similarity.npy'
    cfg.DATASET.D_PROTEIN_WEIGHT_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_protein_weight_similarity.npy'
    cfg.DATASET.D_SE_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_side_effect_relation_onehot.npy'
    cfg.DATASET.D_SE_FREQ_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_side_effect_relation_weight.npy'

    cfg.DATASET.SE_DAG_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_dag_similarity.npy'
    cfg.DATASET.SE_DAG_P_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_dag_p_similarity.npy'
    cfg.DATASET.SE_D_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_drug_relation_onehot.npy'
    cfg.DATASET.SE_D_FREQ_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_drug_relation_weight.npy'
    cfg.DATASET.SE_SIM_LOOKUP = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_lookup.txt'

    cfg.freeze()

    return cfg


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.feed_forward(self.ln2(x))
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.atten_dropout = nn.Dropout(p=dropout)
        self.output_dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        query = self.query(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.key(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.value(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.atten_dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_dropout(self.output_linear(x))


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.dropout(self.w_2(self.dropout(self.activation(self.w_1(x)))))


class SmilesEmbedding(nn.Module):

    def __init__(self, cfg):

        super().__init__()
        self.cfg = cfg

        self.smiles_token_embedding = nn.Embedding(cfg.DATASET.VOCAB_SMILES, cfg.MODEL.HIDDEN, padding_idx=0)
        self.pos_embedding = PositionalEmbedding(cfg.MODEL.HIDDEN, cfg.DATASET.SMILES_LEN)
        self.dropout = nn.Dropout(p=cfg.MODEL.DROPOUT)

    def forward(self, x, pos_num):
        smiles_embedding = self.smiles_token_embedding(x)
        pos_embedding = self.pos_embedding(pos_num)
        final_embedding = smiles_embedding + pos_embedding

        return self.dropout(final_embedding)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class ResidualConv2d(nn.Module):
    def __init__(self, input_channel, output_channel, stride, padding):
        super(ResidualConv2d, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channel),
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_channel),
        )

    def forward(self, x):
        return F.relu(self.conv_block(x) + self.conv_skip(x))


class ResidualConv3d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super(ResidualConv3d, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv3d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(output_channel),
            nn.ReLU(),
            nn.Conv3d(output_channel, output_channel, kernel_size=kernel_size, padding=(1,1,1)),
            nn.BatchNorm3d(output_channel),
        )

        self.conv_skip = nn.Sequential(
            nn.Conv3d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(output_channel),
        )

    def forward(self, x):
        return F.relu(self.conv_block(x) + self.conv_skip(x))


class ResidualCombine(nn.Module):
    def __init__(self, input_channel, output_channel, stride, padding):
        super(ResidualCombine, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride, padding=padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(output_channel),
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride, padding=padding),
            nn.BatchNorm2d(output_channel),
        )

    def forward(self, x):
        return F.relu(self.conv_block(x) + self.conv_skip(x))


class ResidualConv1d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=7, stride=1, padding=1, groups=1):
        super(ResidualConv1d, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),
            nn.Conv1d(output_channel, output_channel, kernel_size=kernel_size, padding=padding, groups=groups),
            nn.BatchNorm1d(output_channel),
        )

        self.conv_skip = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm1d(output_channel),
        )

    def forward(self, x):
        return F.relu(self.conv_block(x) + self.conv_skip(x))


class MDFFHD_regression(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.smiles_transformer = Smiles_Transformer(cfg)
        self.smiles_ln = nn.LayerNorm(cfg.MODEL.HIDDEN)
        self.smiles_w1 = nn.Linear(cfg.MODEL.HIDDEN*100, cfg.MODEL.FINAL_FEATURE_LEN*11)
        self.smiles_activation = nn.GELU()

        self.drug_f1_layer1 = nn.Linear(cfg.DATASET.DRUG_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f1_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f2_layer1 = nn.Linear(cfg.DATASET.DRUG_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f2_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f3_layer1 = nn.Linear(cfg.DATASET.DRUG_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f3_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f4_layer1 = nn.Linear(cfg.DATASET.DRUG_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f4_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f5_layer1 = nn.Linear(cfg.DATASET.DRUG_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f5_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f6_layer1 = nn.Linear(cfg.DATASET.DRUG_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f6_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f7_layer1 = nn.Linear(cfg.DATASET.DRUG_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f7_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)

        self.drug_f1_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f2_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f3_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f4_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f5_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f6_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f7_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)

        self.se_f1_layer1 = nn.Linear(cfg.DATASET.SE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f1_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f2_layer1 = nn.Linear(cfg.DATASET.SE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f2_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f3_layer1 = nn.Linear(cfg.DATASET.SE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f3_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f4_layer1 = nn.Linear(cfg.DATASET.SE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f4_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)

        self.se_f1_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f2_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f3_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f4_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)

        self.dsfdropout = nn.Dropout(cfg.MODEL.DROPOUT)

        self.dsf_activation = nn.ReLU()

        self.d_cnn_2d = nn.Sequential(
            ResidualConv2d(21, 16, 2, 1),
            ResidualConv2d(16, 12, 2, 1),
            ResidualConv2d(12, 8, 2, 1),
            ResidualConv2d(8, 4, 2, 1),
        )

        self.d_cnn_3d = nn.Sequential(
            ResidualConv3d(1, 2, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            ResidualConv3d(2, 2, (3, 3, 3), (2, 2, 2), (0, 1, 1)),
            ResidualConv3d(2, 4, (3, 3, 3), (2, 2, 2), (0, 1, 1)),
            ResidualConv3d(4, 4, (3, 3, 3), (2, 2, 2), (0, 1, 1)),
        )

        self.se_cnn_2d = nn.Sequential(
            ResidualConv2d(6, 3, 2, 1),
            ResidualConv2d(3, 3, 2, 1),
            ResidualConv2d(3, 2, 2, 1),
            ResidualConv2d(2, 2, 2, 1),
        )

        self.se_cnn_3d = nn.Sequential(
            ResidualConv3d(1, 2, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            ResidualConv3d(2, 2, (3, 3, 3), (2, 2, 2), (0, 1, 1)),
            ResidualConv3d(2, 2, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            ResidualConv3d(2, 2, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
        )

        self.combined_conv_2d = nn.Sequential(
            ResidualConv2d(27, 25, 2, 1),
            ResidualConv2d(25, 23, 2, 1),
            ResidualConv2d(23, 21, 2, 1),
            ResidualConv2d(21, 20, 2, 1),
        )

        self.drug_1d_conv = nn.Sequential(
            ResidualConv1d(7, 7, 7, 1, 3, 7),
            ResidualConv1d(7, 7, 7, 1, 3, 7)
        )

        self.se_1d_conv = nn.Sequential(
            ResidualConv1d(4, 4, 7, 1, 3, 4),
            ResidualConv1d(4, 4, 7, 1, 3, 4)
        )

        self.d_w_1 = nn.Linear(cfg.MODEL.FINAL_FEATURE_LEN * 15, cfg.MODEL.FINAL_FEATURE_LEN * 7)
        self.se_w_1 = nn.Linear(cfg.MODEL.FINAL_FEATURE_LEN * 8, cfg.MODEL.FINAL_FEATURE_LEN * 4)
        self.dse_w_1 = nn.Linear(cfg.MODEL.FINAL_FEATURE_LEN * 20, cfg.MODEL.FINAL_FEATURE_LEN * 10)

        self.w_r = nn.Linear(cfg.MODEL.FINAL_FEATURE_LEN*32, 1)
        self.w_c = nn.Linear(cfg.MODEL.FINAL_FEATURE_LEN*32, 5)
        self.final_dropout = nn.Dropout(cfg.MODEL.DROPOUT)
        self.activation = nn.ReLU()

    def forward(self, drug_sim, se_sim, smiles, for_mask, pos_num):
        batch_size = drug_sim.shape[0]
        smiles_features = self.smiles_ln(self.smiles_transformer(smiles, for_mask, pos_num)).view(batch_size, -1)

        drug_f1 = self.drug_f1_layer2(self.dsfdropout(self.dsf_activation(self.drug_f1_bn(self.drug_f1_layer1(drug_sim[:, 0, :])))))
        drug_f2 = self.drug_f2_layer2(self.dsfdropout(self.dsf_activation(self.drug_f2_bn(self.drug_f2_layer1(drug_sim[:, 1, :])))))
        drug_f3 = self.drug_f3_layer2(self.dsfdropout(self.dsf_activation(self.drug_f3_bn(self.drug_f3_layer1(drug_sim[:, 2, :])))))
        drug_f4 = self.drug_f4_layer2(self.dsfdropout(self.dsf_activation(self.drug_f4_bn(self.drug_f4_layer1(drug_sim[:, 3, :])))))
        drug_f5 = self.drug_f5_layer2(self.dsfdropout(self.dsf_activation(self.drug_f5_bn(self.drug_f5_layer1(drug_sim[:, 4, :])))))
        drug_f6 = self.drug_f6_layer2(self.dsfdropout(self.dsf_activation(self.drug_f6_bn(self.drug_f6_layer1(drug_sim[:, 5, :])))))
        drug_f7 = self.drug_f7_layer2(self.dsfdropout(self.dsf_activation(self.drug_f7_bn(self.drug_f7_layer1(drug_sim[:, 6, :])))))

        se_f1 = self.se_f1_layer2(self.dsfdropout(self.dsf_activation(self.se_f1_bn(self.se_f1_layer1(se_sim[:, 0, :])))))
        se_f2 = self.se_f2_layer2(self.dsfdropout(self.dsf_activation(self.se_f2_bn(self.se_f2_layer1(se_sim[:, 1, :])))))
        se_f3 = self.se_f3_layer2(self.dsfdropout(self.dsf_activation(self.se_f3_bn(self.se_f3_layer1(se_sim[:, 2, :])))))
        se_f4 = self.se_f4_layer2(self.dsfdropout(self.dsf_activation(self.se_f4_bn(self.se_f4_layer1(se_sim[:, 3, :])))))

        drug_f_list = [drug_f1, drug_f2, drug_f3, drug_f4, drug_f5, drug_f6, drug_f7]
        se_f_list = [se_f1, se_f2, se_f3, se_f4]

        drug_bmm_list = []
        for i in range(len(drug_f_list)):
            for j in range(i+1, len(drug_f_list)):
                drug_bmm_list.append(torch.bmm(drug_f_list[i].unsqueeze(2), drug_f_list[j].unsqueeze(1)))
        drug_features_images = drug_bmm_list[0].view((-1, 1, self.cfg.MODEL.INITIAL_FEATURE_LEN, self.cfg.MODEL.INITIAL_FEATURE_LEN))
        for i in range(1, len(drug_bmm_list)):
            feature_image = drug_bmm_list[i].view((-1, 1, self.cfg.MODEL.INITIAL_FEATURE_LEN, self.cfg.MODEL.INITIAL_FEATURE_LEN))
            drug_features_images = torch.cat([drug_features_images, feature_image], dim=1)

        se_bmm_list = []
        for i in range(len(se_f_list)):
            for j in range(i+1, len(se_f_list)):
                se_bmm_list.append(torch.bmm(se_f_list[i].unsqueeze(2), se_f_list[j].unsqueeze(1)))
        se_features_images = se_bmm_list[0].view((-1, 1, self.cfg.MODEL.INITIAL_FEATURE_LEN, self.cfg.MODEL.INITIAL_FEATURE_LEN))
        for i in range(1, len(se_bmm_list)):
            feature_image = se_bmm_list[i].view((-1, 1, self.cfg.MODEL.INITIAL_FEATURE_LEN, self.cfg.MODEL.INITIAL_FEATURE_LEN))
            se_features_images = torch.cat([se_features_images, feature_image], dim=1)

        drug_sim_2d = self.d_cnn_2d(drug_features_images)
        drug_sim_3d = self.d_cnn_3d(drug_features_images.unsqueeze(1))
        se_sim_2d = self.se_cnn_2d(se_features_images)
        se_sim_3d = self.se_cnn_3d(se_features_images.unsqueeze(1))

        combined_drug_se_feature_images = torch.cat((drug_features_images, se_features_images), dim=1)
        combined_feature_2d = self.combined_conv_2d(combined_drug_se_feature_images)

        drug_sim_feature = torch.cat((drug_f1.unsqueeze(1), drug_f2.unsqueeze(1), drug_f3.unsqueeze(1), drug_f4.unsqueeze(1),
                                      drug_f5.unsqueeze(1), drug_f6.unsqueeze(1), drug_f7.unsqueeze(1)), dim=1)
        se_sim_feature = torch.cat((se_f1.unsqueeze(1), se_f2.unsqueeze(1), se_f3.unsqueeze(1), se_f4.unsqueeze(1)), dim=1)

        drug_sim_feature_conv1d = self.drug_1d_conv(drug_sim_feature).view(batch_size, -1)
        se_sim_feature_conv1d = self.se_1d_conv(se_sim_feature).view(batch_size, -1)

        d_encoding = torch.cat([drug_sim_2d.view(batch_size, -1), drug_sim_3d.view(batch_size, -1), drug_sim_feature_conv1d.view(batch_size, -1)], dim=1)
        se_encoding = torch.cat([se_sim_2d.view(batch_size, -1), se_sim_3d.view(batch_size, -1), se_sim_feature_conv1d.view(batch_size, -1)], dim=1)

        smiles_encoding = self.final_dropout(self.smiles_activation(self.smiles_w1(smiles_features)))
        d_activation = self.final_dropout(self.activation(self.d_w_1(d_encoding)))
        se_activation = self.final_dropout(self.activation(self.se_w_1(se_encoding)))
        dse_activation = self.final_dropout(self.activation(self.dse_w_1(combined_feature_2d.view(batch_size, -1))))

        final_encoding = torch.cat([smiles_encoding, d_activation, se_activation, dse_activation], dim=1)

        return self.w_r(final_encoding), self.w_c(final_encoding)

    def configure_optimizers(self, cfg):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                           % (str(param_dict.keys() - union_params),)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": cfg.TRAIN.WEIGHT_DECAY},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=cfg.TRAIN.LR)
        return optimizer


class Smiles_Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden = cfg.MODEL.HIDDEN
        self.n_layers = cfg.MODEL.LAYERS
        self.attn_heads = cfg.MODEL.ATTEN_HEAD
        self.feed_forward_hidden = cfg.MODEL.HIDDEN * 4

        self.embedding = SmilesEmbedding(self.cfg)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(cfg.MODEL.HIDDEN, cfg.MODEL.ATTEN_HEAD, cfg.MODEL.HIDDEN * 4, cfg.MODEL.DROPOUT) for _ in range(cfg.MODEL.LAYERS)])

    def forward(self, x, for_mask, pos_num):
        mask = (for_mask > 0).unsqueeze(1).repeat(1, for_mask.size(1), 1).unsqueeze(1)

        x = self.embedding(x, pos_num)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x


def train_regression():
    cfg = set_config_reg()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPUS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    loader_args = dict(batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)
    train_dataset = BasicDataset_reg(cfg)
    train_dataloader = DataLoader(train_dataset, shuffle=True, **loader_args)

    # 创建模型
    net = MDFFHD_regression(cfg)
    net = nn.DataParallel(net).to(device=device)
    net.train()

    optimizer = net.module.configure_optimizers(cfg)
    optim_schedule = ScheduledOptim(optimizer, cfg.TRAIN.LR, n_warmup_steps=cfg.TRAIN.WARMUP_STEPS, final_step=int(len(train_dataset)/cfg.TRAIN.BATCH_SIZE), cfg=cfg)
    criterion_class = nn.CrossEntropyLoss()

    train_loss = 0
    for epoch in tqdm(range(cfg.TRAIN.EPOCHS), position=0, desc="Epoch", colour='green', ncols=100):
        for i, batch_data in enumerate(tqdm(train_dataloader, desc=f'epoch:{epoch}', ncols=100)):
            drug_sim = batch_data['drug_sim'].to(device=device, non_blocking=True)
            se_sim = batch_data['se_sim'].to(device=device, non_blocking=True)
            smiles_index_tensor = batch_data['smiles_index_tensor'].to(device=device, non_blocking=True)
            for_mask = batch_data['for_mask'].to(device=device, non_blocking=True)
            pos_num = batch_data['pos_num'].to(device=device, non_blocking=True)
            true_freq = batch_data['true_freq'].to(device=device, non_blocking=True).unsqueeze(1)
            freq_class_index = batch_data['freq_class_index'].to(device=device, non_blocking=True)

            pred_freq, pred_freq_class = net(drug_sim, se_sim, smiles_index_tensor, for_mask, pos_num)

            loss = mse_loss(pred_freq, true_freq) + l1_loss(pred_freq, true_freq) + \
                   criterion_class(pred_freq_class, freq_class_index)

            optim_schedule.zero_grad()
            loss.backward()
            optim_schedule.step_and_update_lr()
            train_loss += loss.item()

        if epoch == cfg.TRAIN.EPOCHS-1:
            torch.save(net.state_dict(), r'./regression_snapshot.pth')


def select_neg_samples(cfg):
    original_drug_se_path = cfg.DATASET.DRUG_SE_PATH

    se_onto_path = '../data/se_ontology.csv'

    se_onto = pd.read_csv(se_onto_path, usecols=[0, 1]).to_numpy()

    df_original_drug_se = pd.read_csv(original_drug_se_path, usecols=[0, 1, 2, 3], dtype={0:str, 1:str, 2:str, 3:str})
    cids_to_write = df_original_drug_se.iloc[:, 0].tolist()
    se_name_to_write = df_original_drug_se.iloc[:, 1].tolist()
    freq_to_write = df_original_drug_se.iloc[:, 2].tolist()
    onto_to_write = df_original_drug_se.iloc[:, 3].tolist()

    drug_array = np.unique(cids_to_write)
    se_array = np.unique(se_name_to_write)

    freq_matrix = np.zeros((len(drug_array), len(se_array)))

    for pair_idx in range(len(df_original_drug_se)):
        drug_cids = df_original_drug_se.iloc[pair_idx, 0]
        se_name = df_original_drug_se.iloc[pair_idx, 1]
        pair_freq = df_original_drug_se.iloc[pair_idx, 2]

        row_idx = np.where(drug_array == drug_cids)[0][0]
        col_idx = np.where(se_array == se_name)[0][0]
        freq_matrix[row_idx, col_idx] = pair_freq

    k = 0
    interaction_target = np.zeros((freq_matrix.shape[0]*freq_matrix.shape[1], 3)).astype(int)
    for i in range(freq_matrix.shape[0]):
        for j in range(freq_matrix.shape[1]):
            interaction_target[k, 0] = i
            interaction_target[k, 1] = j
            interaction_target[k, 2] = freq_matrix[i, j]
            k = k + 1
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])
    negative_sample = data_shuffle[0:interaction_target.shape[0] - number_positive]
    a = np.arange(interaction_target.shape[0] - number_positive)
    a = list(a)
    shuffle_negative_idx = random.sample(a, (interaction_target.shape[0] - number_positive))
    negative_sample_array = negative_sample[shuffle_negative_idx[0:number_positive], :]

    for neg_idx in range(len(negative_sample_array)):
        nag_info = negative_sample_array[neg_idx]
        neg_drug_cids = drug_array[nag_info[0]]
        neg_se_name = se_array[nag_info[1]]
        neg_freq = nag_info[2]

        neg_se_onto = se_onto[np.where(se_onto[:, 0] == neg_se_name)[0][0], 1]

        cids_to_write.append(neg_drug_cids)
        se_name_to_write.append(neg_se_name)
        freq_to_write.append(neg_freq)
        onto_to_write.append(neg_se_onto)

    dict_to_write = {}
    dict_to_write['CIDS'] = cids_to_write
    dict_to_write['SE_NAME'] = se_name_to_write
    dict_to_write['SE_FREQ'] = freq_to_write
    dict_to_write['SE_ONTOLOGY'] = onto_to_write

    # output path
    csv_path_to_write_path = './drug_se_freq_classification.csv'

    pd.DataFrame(dict_to_write).to_csv(csv_path_to_write_path, index=False)


def set_config_classification():
    # config
    cfg = CN()

    cfg.EXP_NAME = 'classification'
    cfg.GPUS = "'0, 1'"
    cfg.WORKERS = 16
    cfg.PIN_MEMORY = True

    cfg.MODEL = CN()
    cfg.MODEL.HIDDEN = 128
    cfg.MODEL.LAYERS = 3
    cfg.MODEL.ATTEN_HEAD = 4
    cfg.MODEL.DROPOUT = 0.6
    cfg.MODEL.INITIAL_FEATURE_LEN = 256
    cfg.MODEL.FINAL_FEATURE_LEN = 256

    cfg.DATASET = CN()
    cfg.DATASET.VOCAB_ROOT = './data'
    cfg.DATASET.VOCAB_SMILES = 48
    cfg.DATASET.DRUG_INFO_PATH = './data/drug_info.csv'
    cfg.DATASET.DRUG_SE_PATH = './data/drug_se_freq.csv'
    cfg.DATASET.DRUG_SE_CLASS_PATH = './drug_se_freq_classification.csv'
    cfg.DATASET.ON_MEMORY = True
    cfg.DATASET.DRUG_INFO_COL = 'CIDS SMILES'
    cfg.DATASET.DRUG_SE_COL = 'CIDS SE_NAME SE_FREQ SE_ONTOLOGY'
    cfg.DATASET.SMILES_LEN = 100
    cfg.DATASET.SIMILARITY_ROOT_PATH = './data'
    cfg.DATASET.DRUG_LEN = 757
    cfg.DATASET.SE_LEN = 994

    cfg.TRAIN = CN()
    cfg.TRAIN.EPOCHS = 45
    cfg.TRAIN.BATCH_SIZE = 400
    cfg.TRAIN.LR = 0.0001
    cfg.TRAIN.LR_FACTOR = 0.97
    cfg.TRAIN.LR_STEP = 169
    cfg.TRAIN.WEIGHT_DECAY = 0.01
    cfg.TRAIN.WARMUP_STEPS = 15
    cfg.TRAIN.FOLD = 10

    cfg.defrost()

    # drug vocab
    cfg.DATASET.SMILES_VOCAB = cfg.DATASET.VOCAB_ROOT + '/smiles_subword.json'

    # similarity
    cfg.DATASET.D_COMBINE_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_combined_score_similarity.npy'
    cfg.DATASET.D_MORGAN_R1_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_morgan_radius1_similarity.npy'
    cfg.DATASET.D_MORGAN_R2_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_morgan_radius2_similarity.npy'
    cfg.DATASET.D_PROTEIN_ONEHOT_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_protein_onehot_similarity.npy'
    cfg.DATASET.D_PROTEIN_WEIGHT_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_protein_weight_similarity.npy'
    cfg.DATASET.D_SE_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_side_effect_relation_onehot.npy'
    cfg.DATASET.D_SE_FREQ_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_side_effect_relation_weight.npy'

    cfg.DATASET.SE_DAG_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_dag_similarity.npy'
    cfg.DATASET.SE_DAG_P_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_dag_p_similarity.npy'
    cfg.DATASET.SE_D_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_drug_relation_onehot.npy'
    cfg.DATASET.SE_D_FREQ_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_drug_relation_weight.npy'
    cfg.DATASET.SE_SIM_LOOKUP = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_lookup.txt'

    cfg.freeze()

    return cfg


class BasicDataset_classification(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.drug_info = pd.read_csv(cfg.DATASET.DRUG_INFO_PATH, usecols=cfg.DATASET.DRUG_INFO_COL.split(' ')).to_numpy()
        self.drug_se = pd.read_csv(cfg.DATASET.DRUG_SE_CLASS_PATH, usecols=cfg.DATASET.DRUG_SE_COL.split(' ')).to_numpy().astype(str)
        self.smiles_vocab = Vocab.load(cfg.DATASET.SMILES_VOCAB)

        self.d_combined_sim = np.load(cfg.DATASET.D_COMBINE_SIM)
        self.d_morgan_r1_sim = np.load(cfg.DATASET.D_MORGAN_R1_SIM)
        self.d_morgan_r2_sim = np.load(cfg.DATASET.D_MORGAN_R2_SIM)
        self.d_protein_onehot_sim = np.load(cfg.DATASET.D_PROTEIN_ONEHOT_SIM)
        self.d_protein_weight_sim = np.load(cfg.DATASET.D_PROTEIN_WEIGHT_SIM)

        self.se_dag_sim = np.load(cfg.DATASET.SE_DAG_SIM)
        self.se_dag_p_sim = np.load(cfg.DATASET.SE_DAG_P_SIM)

        self.se_sim_lookup = pd.read_csv(cfg.DATASET.SE_SIM_LOOKUP, header=None).iloc[:, 0].to_numpy()
        self.cids_list = self.drug_info[:, 0]

        self.se_d_sim, self.se_d_freq_sim, self.d_se_sim, self.d_se_freq_sim = self._generate_freq_sim()

    def __len__(self):
        return self.drug_se.shape[0]

    def __getitem__(self, idx):
        drug_se_rela = self.drug_se[idx, :]
        drug_se_freq = drug_se_rela[2].astype(np.float)
        smiles = self._get_drug_info(drug_se_rela[0])

        for_mask = []
        smiles_index_list = []
        for i in smiles:
            smiles_index_list.append(self.smiles_vocab[i])
        smiles_index_list = smiles_index_list[:self.cfg.DATASET.SMILES_LEN]
        for_mask.extend([1 for _ in range(len(smiles_index_list))])
        padding = [self.smiles_vocab['<pad>'] for _ in range(self.cfg.DATASET.SMILES_LEN - len(smiles_index_list))]
        for_mask.extend([0 for _ in range(self.cfg.DATASET.SMILES_LEN - len(smiles_index_list))])
        smiles_index_list = smiles_index_list + padding

        drug_sim_array, se_sim_array = self._get_similarity(drug_se_rela[0], drug_se_rela[1])
        bin_class_label = 1 if drug_se_freq > 0 else 0

        output = {'drug_sim':torch.tensor(drug_sim_array, dtype=torch.float),
                  'se_sim':torch.tensor(se_sim_array, dtype=torch.float),
                  'smiles_index_tensor':torch.tensor(smiles_index_list, dtype=torch.long),
                  'for_mask':torch.tensor(for_mask, dtype=torch.long),
                  'pos_num':torch.arange(self.cfg.DATASET.SMILES_LEN),
                  'bin_class_label': torch.tensor(bin_class_label).long(),
                  'true_prob': torch.tensor(bin_class_label).float()
                  }

        return output

    def _get_drug_info(self, cids):
        drug_info_idx = np.where(self.drug_info[:, 0] == cids)[0][0]  # 只有一个对应的，所以最后用0索引
        smiles = self.drug_info[drug_info_idx, 1]

        return smiles

    def _generate_freq_sim(self):
        se_d_relation = np.load(self.cfg.DATASET.SE_D_SIM)
        se_d_freq_relation = np.load(self.cfg.DATASET.SE_D_FREQ_SIM)
        se_d_sim = cosine_similarity(se_d_relation)
        se_d_freq_sim = cosine_similarity(se_d_freq_relation)
        d_se_sim = cosine_similarity(se_d_relation.T)
        d_se_freq_sim = cosine_similarity(se_d_freq_relation.T)
        return se_d_sim, se_d_freq_sim, d_se_sim, d_se_freq_sim

    def _get_similarity(self, cids, se_name):
        cids_idx = np.where(self.cids_list == cids)[0][0]
        se_idx = np.where(self.se_sim_lookup == se_name)[0][0]

        d_se_sim = self.d_se_sim[cids_idx, :]
        d_se_freq_sim = self.d_se_freq_sim[cids_idx, :]
        d_combined_sim = self.d_combined_sim[cids_idx, :]
        d_morgan_r1_sim = self.d_morgan_r1_sim[cids_idx, :]
        d_morgan_r2_sim = self.d_morgan_r2_sim[cids_idx, :]
        d_protein_onehot_sim = self.d_protein_onehot_sim[cids_idx, :]
        d_protein_weight_sim = self.d_protein_weight_sim[cids_idx, :]

        drug_sim_array = np.stack((d_se_sim, d_se_freq_sim, d_combined_sim, d_morgan_r1_sim,
                                   d_morgan_r2_sim, d_protein_onehot_sim, d_protein_weight_sim), axis=0)
        assert drug_sim_array.shape == (7, 757), f'the wrong shape is {drug_sim_array.shape}'

        # side effect similarity
        se_d_sim = self.se_d_sim[se_idx, :]
        se_d_freq_sim = self.se_d_freq_sim[se_idx, :]
        se_dag_sim = self.se_dag_sim[se_idx, :]
        se_dag_p_sim = self.se_dag_p_sim[se_idx, :]

        se_sim_array = np.stack((se_d_sim, se_d_freq_sim, se_dag_sim, se_dag_p_sim), axis=0)
        assert se_sim_array.shape == (4, 994), f'the wrong shape is {se_sim_array.shape}'

        return drug_sim_array, se_sim_array


class MDFFHD_classification(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.smiles_transformer = Smiles_Transformer(cfg)
        self.smiles_ln = nn.LayerNorm(cfg.MODEL.HIDDEN)
        self.smiles_w1 = nn.Linear(cfg.MODEL.HIDDEN*100, cfg.MODEL.FINAL_FEATURE_LEN*11)
        self.smiles_activation = nn.GELU()

        self.drug_f1_layer1 = nn.Linear(cfg.DATASET.DRUG_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f1_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f2_layer1 = nn.Linear(cfg.DATASET.DRUG_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f2_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f3_layer1 = nn.Linear(cfg.DATASET.DRUG_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f3_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f4_layer1 = nn.Linear(cfg.DATASET.DRUG_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f4_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f5_layer1 = nn.Linear(cfg.DATASET.DRUG_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f5_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f6_layer1 = nn.Linear(cfg.DATASET.DRUG_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f6_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f7_layer1 = nn.Linear(cfg.DATASET.DRUG_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f7_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)

        self.drug_f1_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f2_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f3_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f4_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f5_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f6_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.drug_f7_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)

        self.se_f1_layer1 = nn.Linear(cfg.DATASET.SE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f1_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f2_layer1 = nn.Linear(cfg.DATASET.SE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f2_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f3_layer1 = nn.Linear(cfg.DATASET.SE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f3_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f4_layer1 = nn.Linear(cfg.DATASET.SE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f4_layer2 = nn.Linear(cfg.MODEL.INITIAL_FEATURE_LEN, cfg.MODEL.INITIAL_FEATURE_LEN)

        self.se_f1_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f2_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f3_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)
        self.se_f4_bn = nn.BatchNorm1d(cfg.MODEL.INITIAL_FEATURE_LEN)

        self.dsfdropout = nn.Dropout(cfg.MODEL.DROPOUT)

        self.dsf_activation = nn.ReLU()

        self.d_cnn_2d = nn.Sequential(
            ResidualConv2d(21, 16, 2, 1),
            ResidualConv2d(16, 12, 2, 1),
            ResidualConv2d(12, 8, 2, 1),
            ResidualConv2d(8, 4, 2, 1),
        )

        self.d_cnn_3d = nn.Sequential(
            ResidualConv3d(1, 2, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            ResidualConv3d(2, 2, (3, 3, 3), (2, 2, 2), (0, 1, 1)),
            ResidualConv3d(2, 4, (3, 3, 3), (2, 2, 2), (0, 1, 1)),
            ResidualConv3d(4, 4, (3, 3, 3), (2, 2, 2), (0, 1, 1)),
        )

        self.se_cnn_2d = nn.Sequential(
            ResidualConv2d(6, 3, 2, 1),
            ResidualConv2d(3, 3, 2, 1),
            ResidualConv2d(3, 2, 2, 1),
            ResidualConv2d(2, 2, 2, 1),
        )

        self.se_cnn_3d = nn.Sequential(
            ResidualConv3d(1, 2, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            ResidualConv3d(2, 2, (3, 3, 3), (2, 2, 2), (0, 1, 1)),
            ResidualConv3d(2, 2, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            ResidualConv3d(2, 2, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
        )

        self.combined_conv_2d = nn.Sequential(
            ResidualConv2d(27, 25, 2, 1),
            ResidualConv2d(25, 23, 2, 1),
            ResidualConv2d(23, 21, 2, 1),
            ResidualConv2d(21, 20, 2, 1),
        )

        self.drug_1d_conv = nn.Sequential(
            ResidualConv1d(7, 7, 7, 1, 3, 7),
            ResidualConv1d(7, 7, 7, 1, 3, 7)
        )

        self.se_1d_conv = nn.Sequential(
            ResidualConv1d(4, 4, 7, 1, 3, 4),
            ResidualConv1d(4, 4, 7, 1, 3, 4)
        )

        self.d_w_1 = nn.Linear(cfg.MODEL.FINAL_FEATURE_LEN * 15, cfg.MODEL.FINAL_FEATURE_LEN * 7)
        self.se_w_1 = nn.Linear(cfg.MODEL.FINAL_FEATURE_LEN * 8, cfg.MODEL.FINAL_FEATURE_LEN * 4)
        self.dse_w_1 = nn.Linear(cfg.MODEL.FINAL_FEATURE_LEN * 20, cfg.MODEL.FINAL_FEATURE_LEN * 10)

        self.w_c = nn.Linear(cfg.MODEL.FINAL_FEATURE_LEN*32, 2)
        self.final_dropout = nn.Dropout(0.7)
        self.activation = nn.ReLU()

    def forward(self, drug_sim, se_sim, smiles, for_mask, pos_num):
        batch_size = drug_sim.shape[0]

        smiles_features = self.smiles_ln(self.smiles_transformer(smiles, for_mask, pos_num)).view(batch_size, -1)

        drug_f1 = self.drug_f1_layer2(self.dsfdropout(self.dsf_activation(self.drug_f1_bn(self.drug_f1_layer1(drug_sim[:, 0, :])))))
        drug_f2 = self.drug_f2_layer2(self.dsfdropout(self.dsf_activation(self.drug_f2_bn(self.drug_f2_layer1(drug_sim[:, 1, :])))))
        drug_f3 = self.drug_f3_layer2(self.dsfdropout(self.dsf_activation(self.drug_f3_bn(self.drug_f3_layer1(drug_sim[:, 2, :])))))
        drug_f4 = self.drug_f4_layer2(self.dsfdropout(self.dsf_activation(self.drug_f4_bn(self.drug_f4_layer1(drug_sim[:, 3, :])))))
        drug_f5 = self.drug_f5_layer2(self.dsfdropout(self.dsf_activation(self.drug_f5_bn(self.drug_f5_layer1(drug_sim[:, 4, :])))))
        drug_f6 = self.drug_f6_layer2(self.dsfdropout(self.dsf_activation(self.drug_f6_bn(self.drug_f6_layer1(drug_sim[:, 5, :])))))
        drug_f7 = self.drug_f7_layer2(self.dsfdropout(self.dsf_activation(self.drug_f7_bn(self.drug_f7_layer1(drug_sim[:, 6, :])))))

        se_f1 = self.se_f1_layer2(self.dsfdropout(self.dsf_activation(self.se_f1_bn(self.se_f1_layer1(se_sim[:, 0, :])))))
        se_f2 = self.se_f2_layer2(self.dsfdropout(self.dsf_activation(self.se_f2_bn(self.se_f2_layer1(se_sim[:, 1, :])))))
        se_f3 = self.se_f3_layer2(self.dsfdropout(self.dsf_activation(self.se_f3_bn(self.se_f3_layer1(se_sim[:, 2, :])))))
        se_f4 = self.se_f4_layer2(self.dsfdropout(self.dsf_activation(self.se_f4_bn(self.se_f4_layer1(se_sim[:, 3, :])))))

        drug_f_list = [drug_f1, drug_f2, drug_f3, drug_f4, drug_f5, drug_f6, drug_f7]
        se_f_list = [se_f1, se_f2, se_f3, se_f4]

        drug_bmm_list = []
        for i in range(len(drug_f_list)):
            for j in range(i+1, len(drug_f_list)):
                drug_bmm_list.append(torch.bmm(drug_f_list[i].unsqueeze(2), drug_f_list[j].unsqueeze(1)))
        drug_features_images = drug_bmm_list[0].view((-1, 1, self.cfg.MODEL.INITIAL_FEATURE_LEN, self.cfg.MODEL.INITIAL_FEATURE_LEN))
        for i in range(1, len(drug_bmm_list)):
            feature_image = drug_bmm_list[i].view((-1, 1, self.cfg.MODEL.INITIAL_FEATURE_LEN, self.cfg.MODEL.INITIAL_FEATURE_LEN))
            drug_features_images = torch.cat([drug_features_images, feature_image], dim=1)

        se_bmm_list = []
        for i in range(len(se_f_list)):
            for j in range(i+1, len(se_f_list)):
                se_bmm_list.append(torch.bmm(se_f_list[i].unsqueeze(2), se_f_list[j].unsqueeze(1)))
        se_features_images = se_bmm_list[0].view((-1, 1, self.cfg.MODEL.INITIAL_FEATURE_LEN, self.cfg.MODEL.INITIAL_FEATURE_LEN))
        for i in range(1, len(se_bmm_list)):
            feature_image = se_bmm_list[i].view((-1, 1, self.cfg.MODEL.INITIAL_FEATURE_LEN, self.cfg.MODEL.INITIAL_FEATURE_LEN))
            se_features_images = torch.cat([se_features_images, feature_image], dim=1)

        drug_sim_2d = self.d_cnn_2d(drug_features_images)
        drug_sim_3d = self.d_cnn_3d(drug_features_images.unsqueeze(1))
        se_sim_2d = self.se_cnn_2d(se_features_images)
        se_sim_3d = self.se_cnn_3d(se_features_images.unsqueeze(1))

        combined_drug_se_feature_images = torch.cat((drug_features_images, se_features_images), dim=1)
        combined_feature_2d = self.combined_conv_2d(combined_drug_se_feature_images)

        drug_sim_feature = torch.cat((drug_f1.unsqueeze(1), drug_f2.unsqueeze(1), drug_f3.unsqueeze(1), drug_f4.unsqueeze(1),
                                      drug_f5.unsqueeze(1), drug_f6.unsqueeze(1), drug_f7.unsqueeze(1)), dim=1)
        se_sim_feature = torch.cat((se_f1.unsqueeze(1), se_f2.unsqueeze(1), se_f3.unsqueeze(1), se_f4.unsqueeze(1)), dim=1)

        drug_sim_feature_conv1d = self.drug_1d_conv(drug_sim_feature).view(batch_size, -1)
        se_sim_feature_conv1d = self.se_1d_conv(se_sim_feature).view(batch_size, -1)

        d_encoding = torch.cat([drug_sim_2d.view(batch_size, -1), drug_sim_3d.view(batch_size, -1), drug_sim_feature_conv1d.view(batch_size, -1)], dim=1)
        se_encoding = torch.cat([se_sim_2d.view(batch_size, -1), se_sim_3d.view(batch_size, -1), se_sim_feature_conv1d.view(batch_size, -1)], dim=1)

        smiles_encoding = self.final_dropout(self.smiles_activation(self.smiles_w1(smiles_features)))
        d_activation = self.final_dropout(self.activation(self.d_w_1(d_encoding)))
        se_activation = self.final_dropout(self.activation(self.se_w_1(se_encoding)))
        dse_activation = self.final_dropout(self.activation(self.dse_w_1(combined_feature_2d.view(batch_size, -1))))

        final_encoding = torch.cat([smiles_encoding, d_activation, se_activation, dse_activation], dim=1)

        return self.w_c(final_encoding)

    def configure_optimizers(self, cfg):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                           % (str(param_dict.keys() - union_params),)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": cfg.TRAIN.WEIGHT_DECAY},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=cfg.TRAIN.LR)
        return optimizer


def train_classification():
    cfg = set_config_classification()
    select_neg_samples(cfg)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPUS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    loader_args = dict(batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)
    train_dataset = BasicDataset_classification(cfg)
    train_dataloader = DataLoader(train_dataset, shuffle=True, **loader_args)

    net = MDFFHD_classification(cfg)
    net = nn.DataParallel(net).to(device=device)
    net.train()

    optimizer = net.module.configure_optimizers(cfg)
    optim_schedule = ScheduledOptim(optimizer, cfg.TRAIN.LR, n_warmup_steps=cfg.TRAIN.WARMUP_STEPS, final_step=int(len(train_dataset)/cfg.TRAIN.BATCH_SIZE), cfg=cfg)
    criterion_class = nn.CrossEntropyLoss()

    train_loss = 0
    for epoch in tqdm(range(cfg.TRAIN.EPOCHS), position=0, desc="Epoch", colour='green', ncols=100):
        for i, batch_data in enumerate(tqdm(train_dataloader, desc=f'epoch:{epoch}', ncols=100)):
            drug_sim = batch_data['drug_sim'].to(device=device, non_blocking=True)
            se_sim = batch_data['se_sim'].to(device=device, non_blocking=True)
            smiles_index_tensor = batch_data['smiles_index_tensor'].to(device=device, non_blocking=True)
            for_mask = batch_data['for_mask'].to(device=device, non_blocking=True)
            pos_num = batch_data['pos_num'].to(device=device, non_blocking=True)
            bin_class_label = batch_data['bin_class_label'].to(device=device, non_blocking=True)

            pred_class = net(drug_sim, se_sim, smiles_index_tensor, for_mask, pos_num)

            loss = criterion_class(pred_class, bin_class_label)
            optim_schedule.zero_grad()
            loss.backward()
            optim_schedule.step_and_update_lr()
            train_loss += loss.item()

        if epoch == cfg.TRAIN.EPOCHS-1:
            torch.save(net.state_dict(), r'./classification_snapshot.pth')


def set_config_classification_case_study():
    # config
    cfg = CN()

    cfg.EXP_NAME = 'classification'
    cfg.GPUS = "'0, 1'"
    cfg.WORKERS = 16
    cfg.PIN_MEMORY = True

    cfg.MODEL = CN()
    cfg.MODEL.HIDDEN = 128
    cfg.MODEL.LAYERS = 3
    cfg.MODEL.ATTEN_HEAD = 4
    cfg.MODEL.DROPOUT = 0.6
    cfg.MODEL.INITIAL_FEATURE_LEN = 256
    cfg.MODEL.FINAL_FEATURE_LEN = 256

    cfg.DATASET = CN()
    cfg.DATASET.VOCAB_ROOT = './data'
    cfg.DATASET.VOCAB_SMILES = 48
    cfg.DATASET.DRUG_INFO_PATH = './data/drug_info.csv'
    cfg.DATASET.DRUG_SE_PATH = './data/case_study_pairs.csv'
    cfg.DATASET.DRUG_SE_CLASS_PATH = './data/case_study_pairs.csv'
    cfg.DATASET.ON_MEMORY = True
    cfg.DATASET.DRUG_INFO_COL = 'CIDS SMILES'
    cfg.DATASET.DRUG_SE_COL = 'CIDS SE_NAME SE_FREQ SE_ONTOLOGY'
    cfg.DATASET.SMILES_LEN = 100
    cfg.DATASET.SIMILARITY_ROOT_PATH = './data'
    cfg.DATASET.DRUG_LEN = 757
    cfg.DATASET.SE_LEN = 994

    cfg.TRAIN = CN()
    cfg.TRAIN.EPOCHS = 45
    cfg.TRAIN.BATCH_SIZE = 400
    cfg.TRAIN.LR = 0.0001
    cfg.TRAIN.LR_FACTOR = 0.97
    cfg.TRAIN.LR_STEP = 169
    cfg.TRAIN.WEIGHT_DECAY = 0.01
    cfg.TRAIN.WARMUP_STEPS = 15
    cfg.TRAIN.FOLD = 10

    cfg.defrost()

    # drug vocab
    cfg.DATASET.SMILES_VOCAB = cfg.DATASET.VOCAB_ROOT + '/smiles_subword.json'

    # similarity
    cfg.DATASET.D_COMBINE_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_combined_score_similarity.npy'
    cfg.DATASET.D_MORGAN_R1_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_morgan_radius1_similarity.npy'
    cfg.DATASET.D_MORGAN_R2_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_morgan_radius2_similarity.npy'
    cfg.DATASET.D_PROTEIN_ONEHOT_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_protein_onehot_similarity.npy'
    cfg.DATASET.D_PROTEIN_WEIGHT_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_protein_weight_similarity.npy'
    cfg.DATASET.D_SE_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_side_effect_relation_onehot.npy'
    cfg.DATASET.D_SE_FREQ_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_side_effect_relation_weight.npy'

    cfg.DATASET.SE_DAG_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_dag_similarity.npy'
    cfg.DATASET.SE_DAG_P_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_dag_p_similarity.npy'
    cfg.DATASET.SE_D_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_drug_relation_onehot.npy'
    cfg.DATASET.SE_D_FREQ_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_drug_relation_weight.npy'
    cfg.DATASET.SE_SIM_LOOKUP = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_lookup.txt'

    cfg.freeze()

    return cfg


class BasicDataset_classification_case_study(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.drug_info = pd.read_csv(cfg.DATASET.DRUG_INFO_PATH, usecols=cfg.DATASET.DRUG_INFO_COL.split(' ')).to_numpy()
        self.drug_se = pd.read_csv(cfg.DATASET.DRUG_SE_CLASS_PATH, usecols=cfg.DATASET.DRUG_SE_COL.split(' ')).to_numpy().astype(str)
        self.smiles_vocab = Vocab.load(cfg.DATASET.SMILES_VOCAB)

        self.d_combined_sim = np.load(cfg.DATASET.D_COMBINE_SIM)
        self.d_morgan_r1_sim = np.load(cfg.DATASET.D_MORGAN_R1_SIM)
        self.d_morgan_r2_sim = np.load(cfg.DATASET.D_MORGAN_R2_SIM)
        self.d_protein_onehot_sim = np.load(cfg.DATASET.D_PROTEIN_ONEHOT_SIM)
        self.d_protein_weight_sim = np.load(cfg.DATASET.D_PROTEIN_WEIGHT_SIM)

        self.se_dag_sim = np.load(cfg.DATASET.SE_DAG_SIM)
        self.se_dag_p_sim = np.load(cfg.DATASET.SE_DAG_P_SIM)

        self.se_sim_lookup = pd.read_csv(cfg.DATASET.SE_SIM_LOOKUP, header=None).iloc[:, 0].to_numpy()
        self.cids_list = self.drug_info[:, 0]

        self.se_d_sim, self.se_d_freq_sim, self.d_se_sim, self.d_se_freq_sim = self._generate_freq_sim()

    def __len__(self):
        return self.drug_se.shape[0]

    def __getitem__(self, idx):
        drug_se_rela = self.drug_se[idx, :]
        drug_se_freq = drug_se_rela[2].astype(np.float)
        smiles = self._get_drug_info(drug_se_rela[0])

        for_mask = []
        smiles_index_list = []
        for i in smiles:
            smiles_index_list.append(self.smiles_vocab[i])
        smiles_index_list = smiles_index_list[:self.cfg.DATASET.SMILES_LEN]
        for_mask.extend([1 for _ in range(len(smiles_index_list))])
        padding = [self.smiles_vocab['<pad>'] for _ in range(self.cfg.DATASET.SMILES_LEN - len(smiles_index_list))]
        for_mask.extend([0 for _ in range(self.cfg.DATASET.SMILES_LEN - len(smiles_index_list))])
        smiles_index_list = smiles_index_list + padding

        drug_sim_array, se_sim_array = self._get_similarity(drug_se_rela[0], drug_se_rela[1])
        bin_class_label = 1 if drug_se_freq > 0 else 0

        output = {'pair_idx': torch.tensor(idx, dtype=torch.long),
                  'drug_sim':torch.tensor(drug_sim_array, dtype=torch.float),
                  'se_sim':torch.tensor(se_sim_array, dtype=torch.float),
                  'smiles_index_tensor':torch.tensor(smiles_index_list, dtype=torch.long),
                  'for_mask':torch.tensor(for_mask, dtype=torch.long),
                  'pos_num':torch.arange(self.cfg.DATASET.SMILES_LEN),
                  'bin_class_label': torch.tensor(bin_class_label).long(),
                  'true_prob': torch.tensor(bin_class_label).float()
                  }

        return output

    def _get_drug_info(self, cids):
        drug_info_idx = np.where(self.drug_info[:, 0] == cids)[0][0]  # 只有一个对应的，所以最后用0索引
        smiles = self.drug_info[drug_info_idx, 1]

        return smiles

    def _generate_freq_sim(self):
        se_d_relation = np.load(self.cfg.DATASET.SE_D_SIM)
        se_d_freq_relation = np.load(self.cfg.DATASET.SE_D_FREQ_SIM)
        se_d_sim = cosine_similarity(se_d_relation)
        se_d_freq_sim = cosine_similarity(se_d_freq_relation)
        d_se_sim = cosine_similarity(se_d_relation.T)
        d_se_freq_sim = cosine_similarity(se_d_freq_relation.T)
        return se_d_sim, se_d_freq_sim, d_se_sim, d_se_freq_sim

    def _get_similarity(self, cids, se_name):
        cids_idx = np.where(self.cids_list == cids)[0][0]
        se_idx = np.where(self.se_sim_lookup == se_name)[0][0]

        d_se_sim = self.d_se_sim[cids_idx, :]
        d_se_freq_sim = self.d_se_freq_sim[cids_idx, :]
        d_combined_sim = self.d_combined_sim[cids_idx, :]
        d_morgan_r1_sim = self.d_morgan_r1_sim[cids_idx, :]
        d_morgan_r2_sim = self.d_morgan_r2_sim[cids_idx, :]
        d_protein_onehot_sim = self.d_protein_onehot_sim[cids_idx, :]
        d_protein_weight_sim = self.d_protein_weight_sim[cids_idx, :]

        drug_sim_array = np.stack((d_se_sim, d_se_freq_sim, d_combined_sim, d_morgan_r1_sim,
                                   d_morgan_r2_sim, d_protein_onehot_sim, d_protein_weight_sim), axis=0)
        assert drug_sim_array.shape == (7, 757), f'the wrong shape is {drug_sim_array.shape}'

        # side effect similarity
        se_d_sim = self.se_d_sim[se_idx, :]
        se_d_freq_sim = self.se_d_freq_sim[se_idx, :]
        se_dag_sim = self.se_dag_sim[se_idx, :]
        se_dag_p_sim = self.se_dag_p_sim[se_idx, :]

        se_sim_array = np.stack((se_d_sim, se_d_freq_sim, se_dag_sim, se_dag_p_sim), axis=0)
        assert se_sim_array.shape == (4, 994), f'the wrong shape is {se_sim_array.shape}'

        return drug_sim_array, se_sim_array


def case_study_association_prediction():
    cfg = set_config_classification_case_study()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPUS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 加载数据
    loader_args = dict(batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)
    train_dataset = BasicDataset_classification(cfg)
    train_dataloader = DataLoader(train_dataset, shuffle=True, **loader_args)

    drug_se_pair_dname = pd.read_csv(cfg.DATASET.DRUG_SE_CLASS_PATH, usecols=['NAME']).iloc[:, 0].to_numpy().astype(str)
    drug_se_pair_sname = pd.read_csv(cfg.DATASET.DRUG_SE_CLASS_PATH, usecols=['SE_NAME']).iloc[:, 0].to_numpy().astype(str)

    net = MDFFHD_classification(cfg)
    net = nn.DataParallel(net).to(device=device)
    model_to_load = 'classification_snapshot.pth'
    net.load_state_dict(torch.load(model_to_load, map_location=device))

    net.eval()

    m = torch.nn.Softmax(dim=1)
    pred_score_list = []
    pair_drug_name = []
    pair_se_name = []
    for batch_data in train_dataloader:
        pair_idx = batch_data['pair_idx']
        drug_sim = batch_data['drug_sim'].to(device=device, non_blocking=True)
        se_sim = batch_data['se_sim'].to(device=device, non_blocking=True)
        smiles_index_tensor = batch_data['smiles_index_tensor'].to(device=device, non_blocking=True)
        for_mask = batch_data['for_mask'].to(device=device, non_blocking=True)
        pos_num = batch_data['pos_num'].to(device=device, non_blocking=True)
        bin_class_label = batch_data['bin_class_label'].to(device=device, non_blocking=True)

        with torch.no_grad():
            pred_class = net(drug_sim, se_sim, smiles_index_tensor, for_mask, pos_num)
            prob = m(pred_class)
            pair_idx2 = pair_idx.cpu().numpy().tolist()[0]
            prob_temp = prob.cpu().numpy().tolist()[0][1]

            pred_score_list.append(prob_temp)
            pair_drug_name.append(drug_se_pair_dname[pair_idx2])
            pair_se_name.append(drug_se_pair_sname[pair_idx2])

    dict_to_write = {}
    dict_to_write['NAME'] = pair_drug_name
    dict_to_write['SE_NAME'] = pair_se_name
    dict_to_write['pred_score'] = pred_score_list
    pd.DataFrame(dict_to_write).to_csv(r'./class_pred_score.csv', index=False)


class BasicDataset_reg_case_study(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.drug_info = pd.read_csv(cfg.DATASET.DRUG_INFO_PATH, usecols=cfg.DATASET.DRUG_INFO_COL.split(' ')).to_numpy()
        self.drug_se = pd.read_csv(cfg.DATASET.DRUG_SE_PATH, usecols=cfg.DATASET.DRUG_SE_COL.split(' ')).to_numpy().astype(str)

        self.smiles_vocab = Vocab.load(cfg.DATASET.SMILES_VOCAB)

        self.d_combined_sim = np.load(cfg.DATASET.D_COMBINE_SIM)
        self.d_morgan_r1_sim = np.load(cfg.DATASET.D_MORGAN_R1_SIM)
        self.d_morgan_r2_sim = np.load(cfg.DATASET.D_MORGAN_R2_SIM)
        self.d_protein_onehot_sim = np.load(cfg.DATASET.D_PROTEIN_ONEHOT_SIM)
        self.d_protein_weight_sim = np.load(cfg.DATASET.D_PROTEIN_WEIGHT_SIM)

        self.se_dag_sim = np.load(cfg.DATASET.SE_DAG_SIM)
        self.se_dag_p_sim = np.load(cfg.DATASET.SE_DAG_P_SIM)

        self.se_sim_lookup = pd.read_csv(cfg.DATASET.SE_SIM_LOOKUP, header=None).iloc[:, 0].to_numpy()
        self.cids_list = self.drug_info[:,0]
        self.se_d_sim, self.se_d_freq_sim, self.d_se_sim, self.d_se_freq_sim = self._generate_freq_sim()

    def __len__(self):
        return self.drug_se.shape[0]

    def __getitem__(self, idx):
        drug_se_rela = self.drug_se[idx, :]
        drug_se_freq = drug_se_rela[2].astype(np.float)
        smiles = self._get_drug_info(drug_se_rela[0])

        for_mask = []
        smiles_index_list = []
        for i in smiles:
            smiles_index_list.append(self.smiles_vocab[i])
        smiles_index_list = smiles_index_list[:self.cfg.DATASET.SMILES_LEN]
        for_mask.extend([1 for _ in range(len(smiles_index_list))])
        padding = [self.smiles_vocab['<pad>'] for _ in range(self.cfg.DATASET.SMILES_LEN - len(smiles_index_list))]
        for_mask.extend([0 for _ in range(self.cfg.DATASET.SMILES_LEN - len(smiles_index_list))])
        smiles_index_list = smiles_index_list + padding

        drug_sim_array, se_sim_array = self._get_similarity(drug_se_rela[0], drug_se_rela[1])

        output = {'pair_idx': torch.tensor(idx, dtype=torch.long),
                  'drug_sim':torch.tensor(drug_sim_array, dtype=torch.float),
                  'se_sim':torch.tensor(se_sim_array, dtype=torch.float),
                  'smiles_index_tensor':torch.tensor(smiles_index_list, dtype=torch.long),
                  'for_mask':torch.tensor(for_mask, dtype=torch.long),
                  'pos_num':torch.arange(self.cfg.DATASET.SMILES_LEN),
                  'true_freq':torch.tensor(drug_se_freq, dtype=torch.float),
                  'freq_class_index': torch.tensor(drug_se_freq - 1).long()}
        return output

    def _get_drug_info(self, cids):
        drug_info_idx = np.where(self.drug_info[:, 0] == cids)[0][0]
        smiles = self.drug_info[drug_info_idx, 1]

        return smiles

    def _generate_freq_sim(self):
        se_d_relation = np.load(self.cfg.DATASET.SE_D_SIM)
        se_d_freq_relation = np.load(self.cfg.DATASET.SE_D_FREQ_SIM)
        se_d_sim = cosine_similarity(se_d_relation)
        se_d_freq_sim = cosine_similarity(se_d_freq_relation)
        d_se_sim = cosine_similarity(se_d_relation.T)
        d_se_freq_sim = cosine_similarity(se_d_freq_relation.T)
        return se_d_sim, se_d_freq_sim, d_se_sim, d_se_freq_sim

    def _get_similarity(self, cids, se_name):
        cids_idx = np.where(self.cids_list == cids)[0][0]
        se_idx = np.where(self.se_sim_lookup == se_name)[0][0]

        d_se_sim = self.d_se_sim[cids_idx, :]
        d_se_freq_sim = self.d_se_freq_sim[cids_idx, :]
        d_combined_sim = self.d_combined_sim[cids_idx, :]
        d_morgan_r1_sim = self.d_morgan_r1_sim[cids_idx, :]
        d_morgan_r2_sim = self.d_morgan_r2_sim[cids_idx, :]
        d_protein_onehot_sim = self.d_protein_onehot_sim[cids_idx, :]
        d_protein_weight_sim = self.d_protein_weight_sim[cids_idx, :]

        drug_sim_array = np.stack((d_se_sim, d_se_freq_sim, d_combined_sim, d_morgan_r1_sim,
                                   d_morgan_r2_sim, d_protein_onehot_sim, d_protein_weight_sim), axis=0)

        assert drug_sim_array.shape == (7, 757), f'the wrong shape is {drug_sim_array.shape}'

        # side effect similarity
        se_d_sim = self.se_d_sim[se_idx, :]
        se_d_freq_sim = self.se_d_freq_sim[se_idx, :]
        se_dag_sim = self.se_dag_sim[se_idx, :]
        se_dag_p_sim = self.se_dag_p_sim[se_idx, :]

        se_sim_array = np.stack((se_d_sim, se_d_freq_sim, se_dag_sim, se_dag_p_sim), axis=0)
        assert se_sim_array.shape == (4, 994), f'the wrong shape is {se_sim_array.shape}'

        return drug_sim_array, se_sim_array


def set_config_reg_case_study():
    # config
    cfg = CN()

    cfg.EXP_NAME = 'regression'
    cfg.GPUS = "'0, 1'"
    cfg.WORKERS = 16
    cfg.PIN_MEMORY = True

    cfg.MODEL = CN()
    cfg.MODEL.HIDDEN = 128
    cfg.MODEL.LAYERS = 3
    cfg.MODEL.ATTEN_HEAD = 4
    cfg.MODEL.DROPOUT = 0.1
    cfg.MODEL.INITIAL_FEATURE_LEN = 256
    cfg.MODEL.FINAL_FEATURE_LEN = 256

    cfg.DATASET = CN()
    cfg.DATASET.VOCAB_ROOT = './data'
    cfg.DATASET.VOCAB_SMILES = 48
    cfg.DATASET.DRUG_INFO_PATH = './data/drug_info.csv'
    cfg.DATASET.DRUG_SE_PATH = './data/case_study_pairs.csv'
    cfg.DATASET.ON_MEMORY = True
    cfg.DATASET.DRUG_INFO_COL = 'CIDS SMILES'
    cfg.DATASET.DRUG_SE_COL = 'CIDS SE_NAME SE_FREQ SE_ONTOLOGY'
    cfg.DATASET.SMILES_LEN = 100
    cfg.DATASET.SIMILARITY_ROOT_PATH = './data'
    cfg.DATASET.DRUG_LEN = 757
    cfg.DATASET.SE_LEN = 994

    cfg.TRAIN = CN()
    cfg.TRAIN.EPOCHS = 150
    cfg.TRAIN.BATCH_SIZE = 400
    cfg.TRAIN.LR = 0.0001
    cfg.TRAIN.LR_FACTOR = 0.97
    cfg.TRAIN.LR_STEP = 85
    cfg.TRAIN.WEIGHT_DECAY = 0.01
    cfg.TRAIN.WARMUP_STEPS = 10
    cfg.TRAIN.FOLD = 10

    cfg.defrost()

    # drug vocab
    cfg.DATASET.SMILES_VOCAB = cfg.DATASET.VOCAB_ROOT + '/smiles_subword.json'

    # similarity
    cfg.DATASET.D_COMBINE_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_combined_score_similarity.npy'
    cfg.DATASET.D_MORGAN_R1_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_morgan_radius1_similarity.npy'
    cfg.DATASET.D_MORGAN_R2_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_morgan_radius2_similarity.npy'
    cfg.DATASET.D_PROTEIN_ONEHOT_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_protein_onehot_similarity.npy'
    cfg.DATASET.D_PROTEIN_WEIGHT_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_protein_weight_similarity.npy'
    cfg.DATASET.D_SE_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_side_effect_relation_onehot.npy'
    cfg.DATASET.D_SE_FREQ_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/drug_side_effect_relation_weight.npy'

    cfg.DATASET.SE_DAG_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_dag_similarity.npy'
    cfg.DATASET.SE_DAG_P_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_dag_p_similarity.npy'
    cfg.DATASET.SE_D_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_drug_relation_onehot.npy'
    cfg.DATASET.SE_D_FREQ_SIM = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_drug_relation_weight.npy'
    cfg.DATASET.SE_SIM_LOOKUP = cfg.DATASET.SIMILARITY_ROOT_PATH + '/side_effect_lookup.txt'

    cfg.freeze()

    return cfg


def case_study_frequency_prediction():
    cfg = set_config_reg_case_study()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPUS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    loader_args = dict(batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)
    train_dataset = BasicDataset_reg_case_study(cfg)
    train_dataloader = DataLoader(train_dataset, shuffle=True, **loader_args)

    drug_se_pair_dname = pd.read_csv(cfg.DATASET.DRUG_SE_PATH, usecols=['NAME']).iloc[:, 0].to_numpy().astype(str)
    drug_se_pair_sname = pd.read_csv(cfg.DATASET.DRUG_SE_PATH, usecols=['SE_NAME']).iloc[:, 0].to_numpy().astype(str)

    net = MDFFHD_regression(cfg)
    net = nn.DataParallel(net).to(device=device)
    model_to_load = './regression_snapshot.pth'
    net.load_state_dict(torch.load(model_to_load, map_location=device))
    net.eval()

    pair_drug_name = []
    pair_se_name = []
    pred_freq_list = []
    for batch_data in train_dataloader:
        pair_idx = batch_data['pair_idx']
        drug_sim = batch_data['drug_sim'].to(device=device, non_blocking=True)
        se_sim = batch_data['se_sim'].to(device=device, non_blocking=True)
        smiles_index_tensor = batch_data['smiles_index_tensor'].to(device=device, non_blocking=True)
        for_mask = batch_data['for_mask'].to(device=device, non_blocking=True)
        pos_num = batch_data['pos_num'].to(device=device, non_blocking=True)

        with torch.no_grad():
            pred_freq, pred_freq_class = net(drug_sim, se_sim, smiles_index_tensor, for_mask, pos_num)

            pair_idx2 = pair_idx.numpy().tolist()[0]
            pair_drug_name.append(drug_se_pair_dname[pair_idx2])
            pair_se_name.append(drug_se_pair_sname[pair_idx2])
            pred_freq_list.append(pred_freq.cpu().numpy().tolist()[0][0])

    dict_to_write = {}
    dict_to_write['CIDS_NAME'] = pair_drug_name
    dict_to_write['SE_NAME'] = pair_se_name
    dict_to_write['pred_score'] = pred_freq_list
    pd.DataFrame(dict_to_write).to_csv(r'./pred_freq.csv', index=False)


if __name__ == '__main__':

    train_classification()

    train_regression()

    case_study_association_prediction()

    case_study_frequency_prediction()