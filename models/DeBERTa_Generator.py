from turtle import forward
# import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader

from transformers import DebertaTokenizer, DebertaModel
from transformers import get_linear_schedule_with_warmup
from ..data_utils import ABSADataset
from common_config import model_config


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

    def is_logger(self):
        return True

    def forward(self,):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        '''Prepare optimizer and schedule (linear warmup and decay)'''
        pass

    def optimizer_step(self,):
        pass

    def get_tqdm_dict(self):
        pass

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer,
                                    type_path="train",
                                    args=self.args)
        dataloader = DataLoader(train_dataset,
                                batch_size=self.args.train_batch_size,
                                drop_last=True,
                                shuffle=True,
                                num_workers=model_config.num_workers)
        t_total = (
            (len(dataloader.dataset) //
             (self.args.train_batch_size * max(1, len(self.args.n_gpu)))) //
            self.args.gradient_accumulation_steps *
            float(self.args.num_train_epochs))
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total)
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        pass


# tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
# model = DebertaModel.from_pretrained("microsoft/deberta-base")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# print(inputs)
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state
# # print(last_hidden_states)
