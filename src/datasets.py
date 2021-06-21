import torch
import tensorflow as tf
import config
import pandas as pd
import pickle
import io
import json
import os
import config
from torch.utils.data import Dataset

def get_dataloaders(name):
    if name=='imdb':
        return get_imdb_dataloaders()
    if name=='human_numbers':
        return get_human_numbers_dataloaders()


class IMDBDataset(Dataset):
    def __init__(self,reviews,targets):
        self.reviews=reviews
        self.targets=targets

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self,idx):

        review=self.reviews[idx]
        target=self.targets[idx]

        review=torch.tensor(review,dtype=torch.long)
        target=torch.tensor(target,dtype=torch.float)

        item=(review,
              target)

        return item


def get_imdb_dataloaders():
        data=pd.read_csv('data/raw/imdb.csv')
        data=data.sample(frac=1, random_state=42)

        # Loading data
        train=data[:25000]
        xtrain=train['review'].values.tolist()
        ytrain=train['sentiment'].values

        val=data[25000:]
        xval=val['review'].values.tolist()
        yval=val['sentiment'].values

        # Preprocessing
        tokenizer=tf.keras.preprocessing.text.Tokenizer(num_words=3000)
        tokenizer.fit_on_texts(xtrain)

        tokenizer_json = tokenizer.to_json()
        pth=os.path.join('runs',config.RUN_NAME, config.MODEL+'_tok.json')
        with io.open(pth, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

        xtrain_pro=tokenizer.texts_to_sequences(xtrain)
        xtrain_pro=tf.keras.preprocessing.sequence.pad_sequences(xtrain_pro, maxlen=512)

        xval_pro=tokenizer.texts_to_sequences(xval)
        xval_pro=tf.keras.preprocessing.sequence.pad_sequences(xval_pro, maxlen=512)

        ytrain=[1 if y=='positive' else 0 for y in ytrain]
        yval=[1 if y=='positive' else 0 for y in yval]

        # Creating Datasets
        train_ds=IMDBDataset(xtrain_pro, ytrain)
        val_ds=IMDBDataset(xval_pro, yval)

        # Creating DataLoaders
        train_dl=torch.utils.data.DataLoader(
                train_ds,
                batch_size=config.BATCH_SIZE,
                num_workers=config.WORKER_COUNT,
                )

        val_dl=torch.utils.data.DataLoader(
                val_ds,
                batch_size=config.BATCH_SIZE,
                num_workers=config.WORKER_COUNT,
                )

        return train_dl, val_dl



class HumanNumbersDataset(Dataset):

    def __init__(self, toks, sl):

        self.input_toks=[]
        self.target_toks=[]

        # Input is a sequence from our stream of tokens and the 
        # target is the same sequence but one timestep forward
        for i in range(0, len(toks)-sl-1, sl):
            x=toks[i:i+sl]
            y=toks[i+1:i+sl+1]
            
            self.input_toks.append(x)
            self.target_toks.append(y)

    def __len__(self):
        return len(self.input_toks)

    def __getitem__(self, idx):
        input_toks = self.input_toks[idx]
        target_toks = self.target_toks[idx]

        input_toks = torch.tensor(input_toks, dtype=torch.long)
        target_toks = torch.tensor(target_toks, dtype=torch.long)

        item = (input_toks,
                target_toks)

        return item

def get_human_numbers_dataloaders():
    # Loading data
    with open('data/raw/human_numbers.txt', 'r') as f:
        lines = f.readlines()

        data = ' , '.join(line.strip() for line in lines)

    # Preprocessing
    vocab = set(data.split(' '))
    word2idx = {w: i for i, w in enumerate(list(vocab))}

    pth=os.path.join('runs',config.RUN_NAME, config.MODEL+'_tok.json')
    with io.open(pth, 'w', encoding='utf-8') as f:
        f.write(json.dumps(word2idx, ensure_ascii=False))

    data_pro = [word2idx[w] for w in data.split(' ')]
    
    # Splitting data
    cut=int(len(data_pro)*0.8)
    train_pro=data_pro[:cut]
    val_pro=data_pro[cut:]

    # Creating Datasets
    train_ds = HumanNumbersDataset(train_pro, 16)
    val_ds = HumanNumbersDataset(val_pro, 16)

    # Creating DataLoaders
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.WORKER_COUNT,
        drop_last=True
    )

    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.WORKER_COUNT,
        drop_last=True
    )

    return train_dl, val_dl

