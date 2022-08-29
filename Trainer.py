import torch
from torch.optim import Adam
import torch.nn as nn
from dataset.dataLoader import getDataLoader
from network import LSTM, RNN, GRU
import numpy as np
from utils import metrics, cost, safeCreateDir
from vision import plot_acc
import time


class Trainer:
    def __init__(self, args):
        # safeCreateDir('results/')
        self.args = args
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self._init_data()
        self._init_model()

    def _init_data(self):
        data = getDataLoader(self.args)
        self.traindl = data.traindl
        self.valdl = data.valdl
        self.testdl = data.testdl
        self.num_words = data.num_words

    def _init_model(self):
        self.net = None
        if self.args.model == 'lstm':
            self.net = LSTM(self.num_words, 15).to(self.device)
        elif self.args.model == 'rnn':
            self.net = RNN(self.num_words, 15).to(self.device)
        elif self.args.model == 'gru':
            self.net = GRU(self.num_words, 15).to(self.device)

        self.opt = Adam(self.net.parameters(), lr=1e-3, weight_decay=5e-4)
        self.cri = nn.CrossEntropyLoss()

    def save_model(self):
        torch.save(self.net.state_dict(), 'results/{}/{}.pt'.format(self.args.model, self.args.model))

    def load_model(self):
        self.net.load_state_dict(torch.load('results/{}/{}.pt'.format(self.args.model, self.args.model)))

    @torch.no_grad()
    def val(self):
        self.net.eval()
        cur_preds = np.empty(0)
        cur_labels = np.empty(0)
        for batch, (inputs, targets, lengths) in enumerate(self.valdl):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            # lengths = lengths.to(self.device)
            pred = self.net(inputs, lengths)

            cur_preds = np.concatenate([cur_preds, pred.cpu().detach().numpy().argmax(axis=1)])
            cur_labels = np.concatenate([cur_labels, targets.cpu().numpy()])
        acc, precision, f1, recall = metrics(cur_preds, cur_labels)
        self.net.train()
        return acc, precision, f1, recall

    def train(self):
        patten = 'Iter: %d/%d   [===========]  cost: %.2fs  loss: %.4f  acc: %.4f/%.4f'
        train_accs = []
        val_accs = []
        for epoch in range(self.args.epochs):
            cur_preds = np.empty(0)
            cur_labels = np.empty(0)
            cur_loss = 0
            start = time.time()
            for batch, (inputs, targets, lengths) in enumerate(self.traindl):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # lengths = lengths.to(self.device)
                pred = self.net(inputs, lengths)

                loss = self.cri(pred, targets)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                cur_preds = np.concatenate([cur_preds, pred.cpu().detach().numpy().argmax(axis=1)])
                cur_labels = np.concatenate([cur_labels, targets.cpu().numpy()])
                cur_loss += loss.item()
            acc, precision, f1, recall = metrics(cur_preds, cur_labels)
            val_acc, val_precision, val_f1, val_recall = self.val()
            train_accs.append(acc)
            val_accs.append(val_acc)
            end = time.time()
            print(patten % (
                epoch,
                self.args.epochs,
                end - start,
                cur_loss,
                val_acc,
                acc,
                # val_precision,
                # val_f1,
                # val_recall
            ))

        self.save_model()
        plot_acc(train_accs, val_accs, self.args.model)

    @torch.no_grad()
    def test(self):
        print("test {}...".format(self.args.model))
        print('dataset: {}'.format(self.args.dataset))
        self.load_model()
        patten = 'test score:  acc: %.4f   precision: %.4f   f1: %.4f   recall: %.4f'
        self.net.eval()
        cur_preds = np.empty(0)
        cur_labels = np.empty(0)
        for batch, (inputs, targets, lengths) in enumerate(self.testdl):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            # lengths = lengths.to(self.device)
            pred = self.net(inputs, lengths)

            cur_preds = np.concatenate([cur_preds, pred.cpu().detach().numpy().argmax(axis=1)])
            cur_labels = np.concatenate([cur_labels, targets.cpu().numpy()])
        acc, precision, f1, recall = metrics(cur_preds, cur_labels)
        self.net.train()
        print(patten % (
            acc,
            precision,
            f1,
            recall
        ))