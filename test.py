# from transformers import DebertaTokenizer, DebertaModel

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import copy
import re
import time
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, T5Config
from models.t5Constraint import T5ConstrainedGen
# from transformers import pipeline, Trainer
# from transformers import TrainingArguments
# import numpy as np
# import argparse
# # from datasets import load_dataset, load_metric
# from data_utils import data_samples

model_name = 't5-base'

config = T5Config.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

model = T5ConstrainedGen(config)
# model = T5ConstrainedGen.from_pretrained(model_name)

# training
input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
logits = outputs.logits

# inference
input_ids = tokenizer(
    "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
).input_ids  # Batch size 1

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


import spacy

nlp = spacy.load("en_core_web_md")
tokens = nlp("dog cat banana afskfsd")

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)





# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states)


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
# local_time = time.gmtime()


def get_annotated_aope_targets(sents, labels):
    annotated_targets = []
    num_sents = len(sents)
    for i in range(num_sents):
        tuples = labels[i]
        # tup: ([3, 4], [2])
        for tup in tuples:
            ap, op, sent = tup[0], tup[1], tup[2]
            opStr = [sents[i][j] for j in op]
            # multiple OT for one AP
            if '[' in sents[i][ap[0]]:
                if len(ap) == 1:
                    sents[i][ap[0]
                             ] = f"{sents[i][ap[0]][:-1]}, {' '.join(opt)}]"
                else:
                    sents[i][ap[-1]
                             ] = f"{sents[i][ap[-1]][:-1]}, {' '.join(opt)}]"
            else:
                annotation = f"{' '.join(opt)}"
                if len(ap) == 1:
                    sents[i][ap[0]] = f"[{sents[i][ap[0]]}|{annotation}]"
                else:
                    sents[i][ap[0]] = f"[{sents[i][ap[0]]}"
                    sents[i][ap[-1]] = f"{sents[i][ap[-1]]}|{annotation}]"
        annotated_targets.append(sents[i])

    return annotated_targets


def extract_triplets(seq):
    aps = re.findall('\[.*?\]', seq)
    # aps = list(filter(lambda x:len(x.split('|')) ==  3 , aps))
    aps = [ap[1:-1] for ap in aps]
    print(aps)
    triplets = []
    for ap in aps:
        try:
            a, b, c = ap.split('|')
        except ValueError:
            a, b, c = '', '', ''

        # for ASTE
        if b in sentiment_word_list:
            if ',' in c:
                for op in c.split(', '):
                    triplets.append((a, b, op))
            else:
                triplets.append((a, b, c))
        # for TASD
        else:
            if ',' in b:
                for ac in b.split(', '):
                    triplets.append((a, ac, c))
            else:
                triplets.append((a, b, c))

    return triplets

# local_time = time.strftime("%Y%m%d_%H_%M", local_time)
# print(local_time)


# line ="Prefer to order it and pick it up though because I do n't like the servers , one young woman in particular ."
# labels = [([15], [11, 12, 13], 'NEG'), ([18, 19], [11, 12, 13], 'NEG')]
line = 'Leon is an East Village gem : casual but hip , with well prepared basic French bistro fare , good specials , a warm and lively atmosphere '
labels = [([0], [7], 'POS'), ([0], [9], 'POS'), ([20], [19], 'POS'), ([26], [
    23], 'POS'), ([26], [25], 'POS'), ([15, 16, 17], [12, 13], 'POS')]

sents = [line.split()]
labels = [labels]
senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
sentiment_word_list = ['positive', 'negative', 'neutral']
targets = []
# tokenizer.decode

# input_ids = tokenizer(line, return_tensors="pt")

# print(input_ids)
# source_sents = copy.deepcopy(sents)
# targets = []
# num_sents = len(sents)
# targets = get_annotated_aope_targets(sents, labels )

# seq = ' '.join(targets[0])
# print(line)
# print(seq)
# print('--------------------')

# seq = '[Leon|aspect] is an East Village gem : [casual|positive|aspect=Leon, abcdef] but [hip|positive|aspect=Leon] , with [well prepared|positive|aspect=French bistro fare] basic French bistro fare , [good|positive|aspect=specials] specials , a [warm|positive|aspect=atmosphere] and [lively|positive|aspect=atmosphere] atmosphere'
#  seq = ''

# print(extract_triplets(seq))


# Input : Leon is an East Village gem : casual but hip, with well prepared basic French bistro fare, good specials, a warm and lively atmosphere.
# Output: Leon is an East Village gem : [casual|positive|Aspect=Leon] but [hip|positive|Aspect=Leon], with [well prepared|positive|Aspect=French bistro fare] basic French bistro fare, [good|positive|Aspect=specials] specials, a [warm|positive|Aspect=atmosphere] and [lively|positive|Aspect=atmosphere] atmosphere.


# 目前的问题，生成的aspect和opinion 可能不是原文中的表达

# 同时标记 aspect 和 opinion
