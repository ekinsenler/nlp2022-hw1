import os, sys
import pathlib

import numpy as np
from typing import List, Tuple
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from stud.utils import load_torch_embedding_layer, Vocabulary, get_mask, build_pretrain_embedding
from model import Model
import gensim.downloader
from gensim.models import KeyedVectors
import torch.nn.functional as F

curr_dir = pathlib.Path(__file__)
proj_dir = curr_dir.parent.parent.parent
hw1_dir = curr_dir.parent.parent
home_dir = pathlib.Path.home()
#sys.path.append(proj_dir / "model")

data_train_path= proj_dir/'model'/'data'/ 'train.tsv'
data_dev_path=proj_dir/'model'/'data'/'dev.tsv'
model_path = proj_dir/'model'/'state_dict_model.pt'



def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel(device=device)


class RandomBaseline(Model):
    options = [
        (3111, "B-CORP"),
        (3752, "B-CW"),
        (3571, "B-GRP"),
        (4799, "B-LOC"),
        (5397, "B-PER"),
        (2923, "B-PROD"),
        (3111, "I-CORP"),
        (6030, "I-CW"),
        (6467, "I-GRP"),
        (2751, "I-LOC"),
        (6141, "I-PER"),
        (1800, "I-PROD"),
        (203394, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]


class StudentModel(Model, nn.Module):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self, device = 'cpu', n_hidden = 512, word_embed_dim=100, train_path=data_train_path, dev_path=data_dev_path, pretrained_embed = 'glove-wiki-gigaword-100', saved_model_path=model_path):
        super().__init__()
        super(Model, self).__init__()
        self.input_dim = word_embed_dim
        self.vocab = Vocabulary(train_path,dev_path)
        self.saved_model_path = saved_model_path
        weights = gensim.downloader.load(pretrained_embed)
        trimmed_embed = build_pretrain_embedding(weights, self.vocab, embedd_dim = 100)

        self.embeddings = load_torch_embedding_layer(trimmed_embed,padding_idx=0)
        self.device = torch.device('cuda:0') if device == 'cuda:0' else torch.device('cpu')
        self.to(self.device)
        self.n_hidden = n_hidden
        self.dropout = torch.nn.Dropout(0.4)
        self.lstm = nn.LSTM(self.input_dim, self.n_hidden //2, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(n_hidden, len(self.vocab.id2label)) # number of tags + <pad>

    def forward(self,x):
        # transpose:  (seq_len, batch_size) ==> (batch_size, seq_len)
        lengths = [self.len_unpadded(seq) for seq in x]
        embed = self.embeddings(x)  # batch, seq,tag
        word_list = [embed]
        embed = torch.cat(word_list, 2)
        word_represents = self.dropout(embed)

        # pack and unpack are excellent
        embed_pack = torch.nn.utils.rnn.pack_padded_sequence(word_represents, lengths, batch_first=True, enforce_sorted=False)
        hidden = None

        lstm_output_pack, hidden = self.lstm(embed_pack, hidden)
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output_pack)
        lstm_output = lstm_output.transpose(0,1)
        out = self.dropout(lstm_output)
        out = self.hidden2tag(out)

        return out

    def loss_fn(self, outputs, labels):
        batch_size = outputs.size(0)
        max_seq_len = outputs.size(1)
        loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        outputs = outputs.contiguous().view(batch_size * max_seq_len, -1)
        total_loss = loss_function(outputs, labels.contiguous().view(batch_size * max_seq_len))

        return total_loss

    def len_unpadded(self, x):
        """ get unpadded sequence length"""
        def scalar(x): return x.view(-1).data.tolist()[0]
        return next((i for i, j in enumerate(x) if scalar(j) == 0), len(x))

    def init_hidden(self, batch_size=64):
        return (torch.randn(2, batch_size, self.n_hidden // 2),
                torch.randn(2, batch_size, self.n_hidden // 2))

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
            # STUDENT: implement here your predict function
            # remember to respect the same order of tokens!
            self.load_state_dict(torch.load(self.saved_model_path, map_location=self.device))
            self.eval()
            result = []
            for token in tokens:
                token_id = torch.tensor([self.vocab.word_to_id(i) for i in token]).long()
                print(token)
                print(token_id)
                with torch.no_grad():
                    #mask = get_mask(token_id)
                    prediction = self(token_id.unsqueeze(0))  # add dimension to create batch like in RNN notebook
                    dim1, dim2, dim3 = prediction.shape
                    feature_out = prediction.contiguous().view(dim1 * dim2, -1)
                    _, tag_seq = torch.max(feature_out, 1)
                    tag_seq = tag_seq.view(dim1, dim2)
                    #tag_seq = mask.long() * tag_seq
                    print(tag_seq)
                    result.append([self.vocab.id_to_label(i) for i in tag_seq[0]])

            return result


