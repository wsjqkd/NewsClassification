from jieba import lcut
import json
from tqdm import tqdm
from torchtext.vocab import vocab
from collections import OrderedDict, Counter
from torchtext.transforms import VocabTransform
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import torch
from sklearn.preprocessing import LabelEncoder
import scipy.io as io


def dataParse(text, stop_words):
    _, label, _, content, _ = text.split('_!_')
    words = lcut(content)
    words = [i for i in words if not i in stop_words]
    return words, int(label)


def getStopWords():
    file = open('dataset/data/stopWords/baidu_stopwords.txt', 'r')
    words = [i.strip() for i in file.readlines()]
    file.close()
    return words


def getFormatData(dataset='toutiao_cat_data'):
    file = open('dataset/data/raw_data/' + dataset + '.txt', 'r')
    texts = file.readlines()[:50000]
    file.close()
    stop_words = getStopWords()
    all_words = []
    all_labels = []
    for text in tqdm(texts, ncols=90):
        content, label = dataParse(text, stop_words)
        if len(content) <= 0:
            continue
        all_words.append(content)
        all_labels.append(label)

    ws = sum(all_words, [])
    set_ws = Counter(ws)
    keys = sorted(set_ws, key=lambda x: set_ws[x], reverse=True)
    dict_words = dict(zip(keys, list(range(1, len(set_ws) + 1))))
    ordered_dict = OrderedDict(dict_words)
    my_vocab = vocab(ordered_dict, specials=['<UNK>', '<SEP>'])
    vocab_transform = VocabTransform(my_vocab)
    vector = vocab_transform(all_words)
    vector = [torch.tensor(i) for i in vector]
    lengths = [len(i) for i in vector]
    pad_seq = pad_sequence(vector, batch_first=True)
    labelencoder = LabelEncoder()

    labels = labelencoder.fit_transform(all_labels)
    data = pad_seq.numpy()
    num_classses = max(labels) + 1

    data = {'X': data,
            'label': labels,
            'num_classes': num_classses,
            'lengths': lengths,
            'num_words': len(my_vocab)}
    io.savemat('dataset/data/processedData/%s.mat' % dataset, data)
