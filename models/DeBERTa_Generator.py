import pytorch_lightning as pl

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from transformers import AdamW, DebertaTokenizer, DebertaModel
from transformers import get_linear_schedule_with_warmup

# from ..data_utils import ABSADataset
# from common_config import model_config


model_name = "microsoft/deberta-base"


def Tokenizer():
    return DebertaTokenizer.from_pretrained(model_name)


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer,
                       data_dir=args.dataset,
                       data_type=type_path,
                       paradigm=args.paradigm,
                       task=args.task,
                       max_len=args.max_seq_length)
    
# BERT LSTM
# https://blog.csdn.net/zhangtingduo/article/details/108474401

class DeBERTaGenerator(pl.LightningDataModule):
    def __init__(self, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(val_transforms, test_transforms, dims)
        self.tokenizer = Tokenizer()
        self.encoder = DebertaModel.from_pretrained(model_name)
        self.decoder = None
        self.l1 = nn.Linear( 768* 512, 1)
        # decoder
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lang = lang
        self.max_length = max_length
        self.embedding = nn.Embedding(len(self.lang.tok_to_idx), self.embedding_size, padding_idx=0)
        self.embedding.weight.data.normal_(0, 1 / self.embedding_size**0.5)
        self.embedding.weight.data[0, :] = 0.0

        self.attn_W = nn.Linear(self.hidden_size, self.hidden_size)
        self.copy_W = nn.Linear(self.hidden_size, self.hidden_size)

        self.gru = nn.GRU(2 * self.hidden_size + self.embedding.embedding_dim, self.hidden_size, batch_first=True)  # input = (context + selective read size + embedding)
        self.out = nn.Linear(self.hidden_size, len(self.lang.tok_to_idx))

    # def prepare_data(self):
    #     pass

    # def setup(self, ):
    #     pass

    def forward(self, x):
        print(x)
        outputs = self.encoder(x)
        return torch.relu(self.l1(outputs))
        # return self.model(input_ids,)

    def training_step(self, batch, batch_idx):
        lm_labels = batch["target_ids"]
        y_hat = self(batch["source_ids"])
        loss = F.cross_entropy(y_hat, y)
        return loss

    # def training_epoch_end(self, outputs):
    #     pass

    def validation_step(self, batch,   batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    # def validation_epoch_end(self, outputs):
    #     pass

    def configure_optimizers(self):
        '''Prepare optimizer and schedule (linear warmup and decay)'''
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.args.learning_rate,
                                     eps=self.args.adam_epsilon)
        return optimizer

    # def optimizer_step(self,):
    #     pass

    # def get_tqdm_dict(self):
    #     pass

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer,
                                    type_path="train",
                                    args=self.args)
        dataloader = DataLoader(train_dataset,
                                batch_size=self.args.train_batch_size,
                                drop_last=True,
                                shuffle=True,
                                num_workers=model_config.num_workers)
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer,
                                  type_path="dev",
                                  args=self.args)
        dataloader =  DataLoader(val_dataset,
                          batch_size=self.args.eval_batch_size,
                          num_workers=3)
        return dataloader
        

    def is_logger(self):
        return True


