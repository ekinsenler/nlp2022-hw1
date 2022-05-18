from torch.utils.data import Dataset
from utils import Vocabulary
import torch

class NERDataset (Dataset):
    def __init__(self,path_to_file, vocab):
        self.word_lists = []
        self.tag_lists = []
        self.path = path_to_file
        self.vocab = vocab
        with open(path_to_file) as f:
            word_list = []
            tag_list = []
            for line in f:
                if line[0] == '#':
                    pass
                elif line == '\n':
                    self.word_lists.append(word_list)
                    self.tag_lists.append(tag_list)
                    word_list = []
                    tag_list = []
                else:
                    word, tag = line.strip().split("\t")
                    word = word.lower()
                    word_list.append(word)
                    tag_list.append(tag)
            f.close()

    def __len__(self) -> int:
        return len(self.word_lists)

    def __getitem__(self, idx: int):
        text_id = []
        label_id = []
        text = self.word_lists[idx]
        label = self.tag_lists[idx]

        for word in text:
            text_id.append(self.vocab.word_to_id(word))
        text_tensor = torch.tensor(text_id).long()
        for label_ele in label:
            label_id.append(self.vocab.label_to_id(label_ele))
        label_tensor = torch.tensor(label_id).long()

        return {'text': text_tensor, 'label': label_tensor}

#vocab = Vocabulary(train_path="../../model/data/train.tsv", dev_path="../../model/data/dev.tsv")
#a = NERDataset(path_to_file="../../model/data/dev.tsv", vocab=vocab)

#print('Done')