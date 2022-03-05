import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader

from transformers import AdamW, BartForConditionalGeneration, BartTokenizer
from transformers import get_linear_schedule_with_warmup
from data_utils import ABSADataset

model_name = 'facebook/bart-base'
# model_name = 'facebook/bart-large-cnn'


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer,
                       data_dir=args.dataset,
                       data_type=type_path,
                       paradigm=args.paradigm,
                       task=args.task,
                       max_len=args.max_seq_length)


def Tokenizer():
    return BartTokenizer.from_pretrained(model_name)


class BARTFineTuner(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = BartForConditionalGeneration.from_pretrained(
            model_name)
        self.tokenizer = Tokenizer()

    def is_logger(self):
        return True

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    '''
    For masked language modeling (e.g., BertForMaskedLM), 
    the model expects a tensor of dimension (batch_size, seq_length)
    with each value corresponding to the expected label of each individual token: 
    the labels being the token ID for the masked token, 
    and values to be ignored for the rest (usually -100).
    '''

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        # lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        '''
        经过 tokenizer encode 输出编码后的结果，如果句子长度不一致 会padding

        input_ids       输入文本的编码 
        [101, 1188, 1110, 170, 1603, 4954, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        attention_mask  输出哪些位置是有文本的，哪些位置是补的 padding 
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        输入句子对时
        The first sequence, the “context” used for the question, 
        has all its tokens represented by a 0, 
        whereas the second sequence, corresponding to the “question”, 
        has all its tokens represented by a 1.
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        
        During training, both BART and T5 will make the appropriate decoder_input_ids and decoder attention masks internally. 

        Most encoder-decoder models (BART, T5) create their decoder_input_ids on their own from the labels
        '''
        outputs = self(input_ids=batch["source_ids"],
                       attention_mask=batch["source_mask"],
                       labels=lm_labels,
                       decoder_attention_mask=batch['target_mask']
                       )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        # tensorboard_logs = {"train_loss": loss.detach()}
        tensorboard_logs = {"train_loss": loss.detach()}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        # return {
        #     "avg_train_loss": avg_train_loss,
        #     "log": tensorboard_logs,
        #     'progress_bar': tensorboard_logs
        # }

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        # return {
        #     "avg_val_loss": avg_loss,
        #     "log": tensorboard_logs,
        #     'progress_bar': tensorboard_logs
        # }

    def configure_optimizers(self):
        '''Prepare optimizer and schedule (linear warmup and decay)'''
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.learning_rate,
                          eps=self.args.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self,
                       epoch=None,
                       batch_idx=None,
                       optimizer=None,
                       optimizer_idx=None,
                       optimizer_closure=None,
                       on_tpu=None,
                       using_native_amp=None,
                       using_lbfgs=None):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {
            "loss": "{:.4f}".format(self.trainer.avg_loss),
            "lr": self.lr_scheduler.get_last_lr()[-1]
        }
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer,
                                    type_path="train",
                                    args=self.args)
        dataloader = DataLoader(train_dataset,
                                batch_size=self.args.train_batch_size,
                                drop_last=True,
                                shuffle=True,
                                num_workers=3)
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
        val_dataset = get_dataset(tokenizer=self.tokenizer,
                                  type_path="dev",
                                  args=self.args)
        return DataLoader(val_dataset,
                          batch_size=self.args.eval_batch_size,
                          num_workers=3)
