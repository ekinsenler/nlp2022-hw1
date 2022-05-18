from utils import Vocabulary
import gensim.downloader
from gensim.models import KeyedVectors
from os import chdir
import sys, os
import pathlib
from implementation import build_model
import torch
from tqdm import trange
from dataset import NERDataset
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_mask
from torchmetrics.functional import accuracy
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from torch.nn.utils import clip_grad_norm_

curr_dir = pathlib.Path(__file__)
#os.chdir(curr_dir)
proj_dir = curr_dir.parent.parent.parent

data_train_path= proj_dir/"model"/"data"/ "train.tsv"
data_dev_path=proj_dir/"model"/"data"/"dev.tsv"
model_path = proj_dir/"model"/"state_dict_model.pt"
log_dir = proj_dir/"logs"



batch_size=64
num_epoch = 100
lr = 1e-4


# Load embedding layer
#weights = gensim.downloader.load("glove-wiki-gigaword-50")

# Create vocabulary
#vocab = Vocabulary(data_train_path,data_dev_path)

net = build_model('cpu')

prepare_batch = partial(collate_fn, vocab = net.vocab)
train_data = NERDataset(path_to_file=data_train_path, vocab=net.vocab)
train_data_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,collate_fn=prepare_batch, num_workers=0)

val_data = NERDataset(path_to_file=data_dev_path, vocab=net.vocab)
val_data_loader = DataLoader(val_data,batch_size=batch_size,shuffle=True,collate_fn=prepare_batch, num_workers=0)

#net.predict([[0]])

#net.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
writer = SummaryWriter(log_dir)
epoch = 0
best_f1 = -1

for ep in trange(num_epoch):
    ##############
    ###TRAINING###
    ##############
    batch_number = 0
    net.train()
    train_losses = []
    for x, y in train_data_loader:
        batch_number += 1
        net.zero_grad()
        mask = get_mask(x)
        predicts = net(x)
        loss = net.loss_fn(predicts, y)
        train_losses.append(loss)
        writer.add_scalar('train_loss', loss, batch_number)
        loss.backward()
        clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()
    mean_train_loss = sum(train_losses) / len(train_losses)
    writer.add_scalar('mean_train_loss', mean_train_loss, ep)
    ################
    ###EVALUATION###
    ################
    valid_losses = []
    net.eval()
    for x, y in val_data_loader:
        batch_number += 1
        mask = get_mask(x)
        with torch.no_grad():
            predicts = net(x)
        loss = net.loss_fn(predicts, y)
        valid_losses.append(loss)
        writer.add_scalar('valid_loss', loss, batch_number)
    mean_valid_loss = sum(valid_losses) / len(valid_losses)
    writer.add_scalar('mean_valid_loss', mean_valid_loss, ep)

    print('epoch: ' + str(ep) + ' # ' + ' loss: ' + str(mean_train_loss.item()) + ' # ' + '  val loss: ' + str(mean_valid_loss.item())  + ' # '+ '\n')
torch.save(net.state_dict(), model_path)
writer.close()


print("Done")