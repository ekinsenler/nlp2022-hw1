from tqdm import tqdm
import os
import pickle
import torch
from typing import List, Dict
from torch.nn.utils.rnn import pad_sequence
from gensim.models import KeyedVectors
import numpy as np

class Vocabulary(object):
    def __init__(self, train_path, dev_path ):
        self.id2word = []
        self.word2id = {}
        self.id2label = []
        self.label2id = {}
        self.unk_index = -1
        self.pad_index = -1
        self.word_index = 0
        self.label_index = 0

        self.id2word.append('<PAD>')
        self.id2label.append('<PAD>')
        self.pad_index = self.word_index
        self.word2id['<PAD>'] = self.pad_index
        self.label2id['<PAD>'] = self.pad_index

        self.word_index += 1
        self.label_index += 1
        self.id2word.append('<UNK>')
        self.word2id['<UNK>'] = self.word_index
        self.unk_index = self.word_index
        self.word_index += 1

        self.not_seen_words = 0
        self.seen_words = 0

        with open(train_path) as tf:
            for line in tf:
                if line[0] == "#" or line == "\n":
                    pass
                else:
                    word, tag = line.strip().split("\t")
                    if word not in self.word2id:
                        word = word.lower()
                        self.word2id[word] = self.word_index
                        self.id2word.append(word)
                        self.word_index +=1
                    if tag not in self.label2id:
                        self.label2id[tag] = self.label_index
                        self.id2label.append(tag)
                        self.label_index +=1
        with open(dev_path) as df:
            for line in df:
                if line[0] == "#" or line == "\n":
                    pass
                else:
                    word, tag = line.strip().split("\t")
                    if word not in self.word2id:
                        word = word.lower()
                        self.not_seen_words +=1
                        self.word2id[word] = self.word_index
                        self.id2word.append(word)
                        self.word_index +=1
                    else:
                        self.seen_words +=1
                    if tag not in self.label2id:
                        self.label2id[tag] = self.label_index
                        self.id2label.append(tag)
                        self.label_index +=1
        print("Vocabulary constracted with",self.not_seen_words, " not seen words during training, out of ", self.seen_words, " seen words")

    def unk(self):
        return self.unk_index

    def pad(self):
        return self.pad_index

    def size(self):
        return len(self.id2word)

    def word_to_id(self, word):
        if word in self.word2id:
            return self.word2id[word]
        return self.unk()

    def label_to_id(self, label):
        if label in self.label2id:
            return self.label2id[label]
        return self.unk()

    def id_to_word(self, cur_id):
        return self.id2word[cur_id]

    def id_to_label(self, cur_id):
        return self.id2label[cur_id]

    def items(self):
        return self.word2id.items()

def collate_fn(batch: List[Dict], vocab) -> List[Dict]:
    # extract features and labels from batch
    x = [sample["text"] for sample in batch]
    y = [sample["label"] for sample in batch]
    # convert words to index
    #x = [[vocab.word_to_id(word) for word in sample] for sample in x]
    # convert labels to index
    #y = [[vocab.label_to_id(label) for label in sample] for sample in y]
    # convert features to tensor and pad them
    x = pad_sequence(
        [torch.as_tensor(sample) for sample in x],
        batch_first=True,
        padding_value=vocab.pad()
    )
    # convert and pad labels too
    y = pad_sequence(
        [torch.as_tensor(sample) for sample in y],
        batch_first=True,
        padding_value=vocab.pad()
    )
    return (x, y)

def create_embeddings(embed_path, saved_embed_path):
    embedding_vector = dict()
    if (os.path.exists(saved_embed_path)):
        with open(saved_embed_path, 'rb') as handle:
            embedding_vector = pickle.load(handle)
    else:
        f = open(embed_path, encoding='utf8')
        for line in tqdm(f):
            value = line.split(' ')
            word = value[0]
            vector = torch.tensor([float(c) for c in value[1:]])
            embedding_vector[word] = vector
        with open(saved_embed_path, 'wb') as handle:
            pickle.dump(embedding_vector, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return embedding_vector

def load_torch_embedding_layer(weights: KeyedVectors, padding_idx: int = 0, freeze: bool = False):
    vectors = weights
    # random vector for pad
    pad = np.random.rand(1, vectors.shape[1])
    print(pad.shape)
    # mean vector for unknowns
    unk = np.mean(vectors, axis=0, keepdims=True)
    print(unk.shape)
    # concatenate pad and unk vectors on top of pre-trained weights
    vectors = np.concatenate((pad, unk, vectors))
    # convert to pytorch tensor
    vectors = torch.FloatTensor(vectors)
    # and return the embedding layer
    return torch.nn.Embedding.from_pretrained(vectors, padding_idx=padding_idx, freeze=freeze)

def get_mask(batch_tensor):
    mask = batch_tensor.eq(0)
    mask = mask.eq(0)
    return mask

def build_pretrain_embedding(embed, word_vocab, embedd_dim=100):
    vocab_size = word_vocab.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_vocab.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_vocab.items():
        if word in embed:
            pretrain_emb[index, :] = embed[word]
            perfect_match += 1
        elif word.lower() in embed:
            pretrain_emb[index, :] = embed[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

    pretrain_emb[0, :] = np.zeros((1, embedd_dim))
    pretrained_size = len(embed)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / vocab_size))
    return pretrain_emb