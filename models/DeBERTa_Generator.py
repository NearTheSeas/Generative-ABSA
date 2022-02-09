import pytorch_lightning as pl

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
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
        # decoder
        # num_layers=rnn_layers, bidirectional=True,
        
        self.rnn_layers = 1
        self.hidden_size = 768
        self.embedding_size = 512
        '''
        LSTM
        input_size：x的特征维度
        hidden_size：隐藏层的特征维度
        num_layers：lstm隐层的层数，默认为1
        bias：False则bih=0和bhh=0. 默认为True
        batch_first：True则输入输出的数据格式为 (batch, seq, feature)
        dropout：除最后一层，每一层的输出都进行dropout，默认为: 0
        bidirectional：True则为双向lstm默认为False
        输入：input, (h0, c0)
        输出：output, (hn,cn)
        ''' 
        self.lstm = nn.LSTM(self.embedding_size, 
                            self.hidden_size, 
                            # num_layers=self.rnn_layers, 
                            bidirectional=True, 
                            # dropout=dropout_ratio, 
                            batch_first=True)
    
    def rand_init_hidden(self, batch_size):
        """
        random initialize hidden variable
        """
        return Variable(torch.randn(2 * self.rnn_layers, batch_size, self.hidden_size)), \
            Variable(torch.randn(2 * self.rnn_layers, batch_size, self.hidden_size))

    # def prepare_data(self):
    #     pass

    # def setup(self, ):
    #     pass

    def forward(self, sentences):
        batch_size = sentences.size(0)
        seq_length = sentences.size(1)
        embeds= self.encoder(sentences)
        print(embeds)
        '''
        lstm
        输入数据格式：
        input( batch, seq_len,input_size)
        h0(num_layers * num_directions, batch, hidden_size)
        c0(num_layers * num_directions, batch, hidden_size)
        
        输出数据格式：
        output(seq_len, batch, hidden_size * num_directions)
        hn(num_layers * num_directions, batch, hidden_size)
        cn(num_layers * num_directions, batch, hidden_size)
        '''
        hidden = self.rand_init_hidden(batch_size)
        lstm_out, hidden, _ = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim*2)
        d_lstm_out = self.dropout1(lstm_out)
        l_out = self.liner(d_lstm_out)
        lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)
        return lstm_feats

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


