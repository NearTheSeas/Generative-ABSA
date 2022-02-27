# from transformers import DebertaTokenizer, DebertaModel

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
# from transformers import pipeline, Trainer
# from transformers import TrainingArguments
# import numpy as np
# import argparse
# # from datasets import load_dataset, load_metric
# from data_utils import data_samples

# tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
# model = DebertaModel.from_pretrained("microsoft/deberta-base")

# tokenizer = AutoTokenizer.from_pretrained("t5-base")
# model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")


# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)


# 数据处理
# dataset = load_dataset("yelp_review_full")
# tokenized_datasets = dataset.map(tokenize_function, batched=True)


# # Specify where to save the checkpoints from your training:
# training_args = TrainingArguments(
#     output_dir="test_trainer", evaluation_strategy="epoch")


# # pass Trainer a function to compute and report metrics.
# metric = load_metric("accuracy")


# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     compute_metrics=compute_metrics,
# )

# trainer.train()

# generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

# generator("Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone")
# [{'generated_text':
#     'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, Seven for the Dragon-lords (for them to rule in a world ruled by their rulers, and all who live within the realm'}]

# # ----------------
# inputs = tokenizer(
#     "translate English to German: Hugging Face is a technology company based in New York and Paris",
#     return_tensors="pt",
# )
# outputs = model.generate(
#     inputs["input_ids"], max_length=40, num_beams=4, early_stopping=True)


# # inputs = tokenizer(, return_tensors="pt")
# inputs = tokenizer.batch_encode_plus(["Hello, my dog is cute"],
#                                      padding='max_length',
#                                      truncation=True,
#                                      return_tensors="pt")
# # print(inputs)
# # inputs:  { input_ids  'token_type_ids  attention_mask }
# outputs = model(**inputs)
# print(outputs.last_hidden_state.shape)
# # outputs：{ last_hidden_state grad_fn=<AddBackward0>) hidden_states=None  attentions=None }
# # last_hidden_state.shape  [1, 512, 768] batchSize sentenceSize hiddenSize

# # last_hidden_states = outputs.last_hidden_state
# # print(last_hidden_states)

# model = BERT_LSTM_CRF(config.bert_path, tagset_size, config.bert_embedding, config.rnn_hidden, config.rnn_layer,
#                       dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda)

# seq = 'Target:food, Opinion:cheap, Sentiment:positive; Target:waiters, Opinion:nice, Sentiment:positive'
# all_pt = seq.split('; ')
# for pt in all_pt:
#             try:
#                 a, b, c = [ item.split(':')[1] for item in  pt.split(', ')]
#             except ValueError:
#                 a, b, c = '', '', ''
#             print(a,b,c)

