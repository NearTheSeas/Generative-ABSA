# from transformers import DebertaTokenizer, DebertaModel

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import copy
import re
import time
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, T5Config
from models.t5Constraint import T5ConstrainedGen
from transformers import pipeline, Trainer
from transformers import TrainingArguments
import numpy as np
import torch
import argparse
# from datasets import load_dataset, load_metric
from data_utils import data_samples

model_name = 't5-base'

config = T5Config.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# model = T5ConstrainedGen(config)
model = T5ConstrainedGen.from_pretrained(model_name)

# training
# input_ids = tokenizer("The owner is belligerent to guests that have a complaint.", return_tensors="pt").input_ids
# labels = tokenizer("The [owner|aspect] is [belligerent|negative|aspect=owner] to guests that have a complaint.", return_tensors="pt").input_ids
# loss = model(input_ids=input_ids, labels=labels).loss

#  "The owner is belligerent to guests that have a complaint."
sentence = "On a recent trip , our waiter was extremely dismissive , while no less than three staff members waited hand-and-foot on a pair of Japanese girls seated nearby ."

# inference
input_ids = tokenizer(
   sentence, return_tensors="pt"
).input_ids  # Batch size 1

# print(input_ids)

force_word = "scared"
# force_flexible = ["scream", "screams", "screaming", "screamed"]
force_flexible = ['The', 'food', 'is', 'very', 'average', '...', 'the', 'Thai', 'fusion', 'stuff', 'is', 'a', 'bit', 'too', 'sweet', ',', 'every', 'thing', 'they', 'serve', 'is', 'too', 'sweet', 'here', '.']

force_words_ids = [
    tokenizer([force_word], ).input_ids,
    tokenizer(force_flexible, ).input_ids,
]

# force_words_ids [[[16411]], [[3, 7, 11763], [3, 7, 11763, 7], [25471], [3, 7, 11763, 15, 26]]]


# print(force_words_ids)


# tags = "[ ] | opinion aspect positive neutral negative"
# input_ids2 = tokenizer(
#     tags, return_tensors="pt"
# ).input_ids

# constraint_token_ids =[[15672, 7, 310, 207, 26297, 3, 5, 1], [933, 8, 2630, 17622, 68, 9289, 3, 5, 1], [1862, 8250, 3, 21416, 3, 9052, 109, 15, 19, 3, 9, 398, 3, 55, 1], [94, 65, 248, 26297, 11, 237, 394, 313, 3, 5, 1], [37, 1297, 871, 47, 2033, 3, 21007, 7360, 1014, 11, 3, 15443, 12, 82, 334, 174, 3, 5, 1], [37, 2527, 19, 12815, 4424, 295, 12, 2554, 24, 43, 3, 9, 10394, 3, 5, 1], [1804, 542, 3, 55, 1], [100, 19, 3, 9, 248, 286, 12, 129, 3, 9, 3326, 3506, 3, 5, 1], [37, 871, 19, 1134, 2609, 3, 5, 1], [37, 12909, 10631, 33, 248, 3, 55, 1], [37, 7490, 3832, 2373, 26, 114, 19885, 2093, 103, 4796, 7, 11, 8, 8591, 22962, 3, 5319, 210, 29, 7, 3, 6, 8, 360, 79, 700, 428, 25, 3, 233, 3, 5, 5, 3321, 15, 59, 207, 3, 5, 1], [8012, 30539, 3, 6, 68, 1385, 147, 4094, 286, 3, 5, 1], [11287, 7, 17, 1387, 27, 6381, 10118, 1], [6656, 24, 3, 7, 144, 16, 8, 223, 1067, 4686, 24, 34, 47, 8, 6025, 313, 62, 141, 664, 1204, 3, 5, 1], [421, 1749, 49, 47, 529, 18, 13957, 11, 227, 69, 542, 2031, 4363, 147, 46, 1781, 227, 62, 5563, 3, 6, 62, 130, 59, 787, 136, 387, 42, 3, 76, 324, 7, 1558, 3, 5, 1], [27, 25932, 12, 8, 2743, 3, 6, 68, 3, 88, 47, 59, 237, 3, 9521, 2152, 7578, 3, 5, 1], [31440, 4338, 3139, 3, 55, 1], [3, 104, 27, 1385, 1568, 2133, 32, 5950, 32, 3, 5, 1], [27, 183, 78, 1095, 12, 43, 3, 9, 1627, 4338, 2062, 16, 82, 5353, 3, 5, 1], [37, 2013, 570, 19, 1627, 11, 8, 542, 5607, 7, 140, 13, 82, 1100, 1469, 12, 5308, 3, 5, 1], [27, 333, 48, 2062, 1], [3, 104, 27, 56, 470, 2612, 8, 1237, 3506, 3, 6, 313, 3, 6, 11, 3, 15145, 27, 351, 44, 48, 2062, 3, 5, 1], [37, 2013, 570, 19, 3904, 11, 3616, 11, 2399, 3, 6, 8, 542, 19, 66, 3904, 11, 8, 871, 47, 66, 182, 1245, 3, 6, 207, 44, 70, 2476, 11, 1543, 26, 3, 5, 1], [37, 542, 47, 248, 3, 55, 1], [3, 104, 86, 3, 9, 1246, 13, 28351, 583, 3753, 16, 3661, 3, 6, 165, 1245, 12, 217, 3, 9, 286, 24, 22508, 24, 4166, 3, 6, 11, 131, 6080, 9250, 306, 463, 542, 11, 207, 313, 3, 6, 1059, 3, 5, 1], [100, 19, 8, 286, 12, 4839, 11, 777, 8, 6842, 463, 542, 8, 681, 54, 462, 3, 5, 1], [1336, 14548, 3, 18, 165, 490, 542, 21, 151, 113, 333, 8, 200, 3, 5, 1], [27, 6528, 8, 4643, 182, 231, 68, 8, 542, 47, 59, 1494, 8, 594, 3, 5, 1], [27, 164, 59, 36, 3, 9, 26297, 3, 21184, 68, 27, 54, 817, 25, 24, 8, 542, 270, 19, 131, 8957, 11, 24, 132, 19, 59, 231, 1307, 12, 34, 3, 5, 1], [16869, 19, 396, 2192, 3, 6, 5240, 9, 47, 3, 29, 31, 17, 78, 1434, 893, 3, 5, 1], [27, 43, 16929, 270, 386, 648, 11, 43, 435, 8, 463, 11, 1196, 13, 8, 2495, 12, 36, 1287, 3, 5, 1], [611, 3, 6, 8, 701, 11, 313, 33, 321, 20215, 16914, 3, 5, 1]]
# for ids in constraint_token_ids:
#     tmp =  [784,    3,  908, 1820, 3474, 2663, 1465, 7163, 2841]+ids
#     print(tmp)
    
# constraint_token_ids = [[784,    3,  908, 1820, 3474, 2663, 1465, 7163, 2841]+ids for ids in constraint_token_ids]
# print(constraint_token_ids)
# input_ids3 = [list(x for x in ids if x!=0 ) for ids in constraint_token_ids]


# print(input_ids2)
# batch_size = 10
# tmp = [[784,    3,  908, 1820, 3474, 2663, 1465, 7163, 2841] for _ in range(batch_size)]
# print(tmp)
# tmp = torch.tensor( tmp)
# print(tmp)

# force_words_ids = [torch.cat([ torch.tensor([ 784,    3,  908, 1820, 3474, 2663, 1465, 7163, 2841, ]), input_id], -1).tolist() for input_id in input_ids]
# print(force_words_ids)

# ends =  torch.cat([input_ids,input2] ,-1 )
# print(ends)
# #    outs = model.model.generate(
# #             input_ids=batch['source_ids'].to(device),
# #             attention_mask=batch['source_mask'].to(device),
# #             max_length=128)

# # input_ids tensor([[   37,  2527, 19, 12815,  4424,295, 12,  2554, 24, 43,  3,  9, 10394,  5,  1]])

# print(input_ids)
outputs = model.generate(input_ids, )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# import spacy

# nlp = spacy.load("en_core_web_md")
# tokens = nlp("dog cat banana afskfsd")

# for token in tokens:
#     print(token.text, token.has_vector, token.vector_norm, token.is_oov)


# nlp = spacy.load("en_core_web_md")  # make sure to use larger package!
# doc1 = nlp("I like salty fries and hamburgers.")
# doc2 = nlp("Fast food tastes very good.")

# print(doc1, "<->", doc2, doc1.similarity(doc2))
# french_fries = doc1[2:4]
# burgers = doc1[5]
# print(french_fries, "<->", burgers, french_fries.similarity(burgers))


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
