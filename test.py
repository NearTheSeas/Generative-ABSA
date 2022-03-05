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
import time
import re
import copy
# local_time = time.gmtime()

# local_time = time.strftime("%Y%m%d_%H_%M", local_time)
# print(local_time)

# line ="Prefer to order it and pick it up though because I do n't like the servers , one young woman in particular ."
# labels = [([15], [11, 12, 13], 'NEG'), ([18, 19], [11, 12, 13], 'NEG')]
# line = 'Leon is an East Village gem : casual but hip , with well prepared basic French bistro fare , good specials , a warm and lively atmosphere '
# labels = [([0], [7], 'POS'), ([0], [9], 'POS'), ([20], [19], 'POS'), ([26], [
#     23], 'POS'), ([26], [25], 'POS'), ([15, 16, 17], [12, 13], 'POS')]

# sents = [line.split()]
# labels = [labels]
# senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}

# annotated_targets = []
# num_sents = len(sents)
# source_sents = copy.deepcopy(sents)
# for i in range(num_sents):
#     tuples = labels[i]
#     # tup: ([2], [5], 'NEG')
#     for tup in tuples:
#         ap, op, sent = tup[0], tup[1], tup[2]
#         apStr = [source_sents[i][j] for j in ap]
#         opStr = [source_sents[i][j] for j in op]
#         # if '<Aspect>' in sents[i][ap[0]]:
#         #     pass
#         # else:
#         #     if len(ap) == 1:
#         #         sents[i][ap[0]] = f"<Aspect>{sents[i][ap[0]][:-1]}</Aspect>"
#         #     else:
#         #         sents[i][ap[0]] = f"<Aspect>{sents[i][ap[0]][:-1]}"
#         #         sents[i][ap[-1]] = f"{sents[i][ap[-1]][:-1]}</Aspect>"
#         if '[' in sents[i][ap[0]]:
#             # if len(ap) == 1:
#             #     sents[i][ap[0]] = f"{sents[i][ap[0]][:-1]}, {' '.join(opStr)}]"
#             # else:
#             #     sents[i][ap[-1]] = f"{sents[i][ap[-1]][:-1]}, {' '.join(opStr)}]"
#             pass
#         else:
#             if len(ap) == 1:
#                 sents[i][ap[0]] = f"[{sents[i][ap[0]]}|aspect]"
#             else:
#                 sents[i][ap[0]] = f"[{sents[i][ap[0]]}"
#                 sents[i][ap[-1]] = f"{sents[i][ap[-1]]}|aspect]"

#         # annotation = f"opinion|aspect={' '.join(apStr)}|{senttag2word[sent]}"
#         annotation = f"aspect={' '.join(apStr)}|{senttag2word[sent]}"
#         if '[' in sents[i][op[0]]:
#             if len(op) == 1:
#                 sents[i][op[0]] = f"[{sents[i][op[0]][:-1]}, {' '.join(apStr)}]"
#             else:
#                 sents[i][op[-1]] = f"{sents[i][op[-1]][:-1]}, {' '.join(apStr)}]"
#         else:
#             if len(op) == 1:
#                 sents[i][op[0]] = f"[{sents[i][op[0]]}|{annotation}]"
#             else:
#                 sents[i][op[0]] = f"[{sents[i][op[0]]}"
#                 sents[i][op[-1]] = f"{sents[i][op[-1]]}|{annotation}]"
#     annotated_targets.append(sents[i])
# seq = ' '.join(annotated_targets[0])
# print(line)
# print(seq)
# print('--------------------')

sentiment_word_list = ['positive', 'negative', 'neutral']
seq = '[Leon|aspect] is an East Village gem : [casual|positive|aspect=Leon, abcdef] but [hip|positive|aspect=Leon] , with [well prepared|positive|aspect=French bistro fare] basic French bistro fare , [good|positive|aspect=specials] specials , a [warm|positive|aspect=atmosphere] and [lively|positive|aspect=atmosphere] atmosphere'

def extract_triplets_prompt(seq):
    ops = re.findall('\[.*?\]', seq)
    ops = list(filter(lambda x:len(x.split('|')) ==  3 , ops))
    ops = [ap[1:-1] for ap in ops]
    print(ops)
    triplets = []
    for op in ops:
        try:
            a, b, c = op.split('|')
        except ValueError:
            a, b, c = '', '', ''
        # for ASTE
        if b in sentiment_word_list:
            if '=' in c:
                aspects = c.split('=')[1]
                aspects = aspects.split(', ')
                for item in aspects:
                    triplets.append((a, b, item))
            # if ',' in c:
            #     aspects = c.split(', ')
            #     for item in aspects:
            #         triplets.append((a, b, item))
            # else:
            #     triplets.append((a, b, c))
        # for TASD
        else:
            if ',' in b:
                for ac in b.split(', '):
                    triplets.append((a, ac, c))
            else:
                triplets.append((a, b, c))
    return triplets

#  seq = ''

print(extract_triplets_prompt(seq))


# Input : Leon is an East Village gem : casual but hip, with well prepared basic French bistro fare, good specials, a warm and lively atmosphere.
# Output: Leon is an East Village gem : [casual|positive|Aspect=Leon] but [hip|positive|Aspect=Leon], with [well prepared|positive|Aspect=French bistro fare] basic French bistro fare, [good|positive|Aspect=specials] specials, a [warm|positive|Aspect=atmosphere] and [lively|positive|Aspect=atmosphere] atmosphere.


# 目前的问题，生成的aspect和opinion 可能不是原文中的表达

# 同时标记 aspect 和 opinion
