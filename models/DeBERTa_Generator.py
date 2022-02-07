import pytorch_lightning as pl

import torch
import torch.nn.functional as F
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


class DeBERTaGenerator(pl.LightningDataModule):
    def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        self.tokenizer = Tokenizer()
        self.encoder = DebertaModel.from_pretrained(model_name)
        self.decoder = None
        self.l1 = nn.Linear(28 * 28, 10)

    # def prepare_data(self):
    #     pass

    # def setup(self, ):
    #     pass

    def forward(self, x):
        print(x)
        outputs = self.encoder(x)
        return torch.relu(self.l1(x.view(x.size(0), -1)))
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


