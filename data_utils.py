# This file contains all data loading and transformation functions
import os
import random
from utils import tokens_to_seq, contains_digit
from operator import itemgetter
import time
import copy
from torch.utils.data import Dataset

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}


def data_samples(args, tokenizer, data_type):
    # show one sample to check the sanity of the code and the expected output
    print(
        f"Here is an example (from dev set) under `{args.paradigm}` paradigm:")
    dataset = ABSADataset(tokenizer=tokenizer,
                          data_dir=args.dataset,
                          data_type=data_type,
                          paradigm=args.paradigm,
                          task=args.task,
                          max_len=args.max_seq_length)
    data_sample = dataset[random.randint(1, 100)]  # a random data sample
    print('Input :',
          tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
    print('Output:',
          tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))


def read_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    print(f"Total examples = {len(sents)}")
    return sents, labels


def get_annotated_uabsa_targets(sents, labels):
    annotated_targets = []
    num_sents = len(sents)
    for i in range(num_sents):
        tuples = labels[i]
        if tuples != []:
            # tup: ([3, 4], POS)
            for tup in tuples:
                ap, sent = tup[0], tup[1]
                if len(ap) == 1:
                    sents[i][ap[0]
                             ] = f"[{sents[i][ap[0]]}|{senttag2word[sent]}]"
                else:
                    sents[i][ap[0]] = f"[{sents[i][ap[0]]}"
                    sents[i][ap[-1]
                             ] = f"{sents[i][ap[-1]]}|{senttag2word[sent]}]"
        annotated_targets.append(sents[i])

    return annotated_targets


def get_annotated_aope_targets(sents, labels):
    annotated_targets = []
    num_sents = len(sents)
    for i in range(num_sents):
        tuples = labels[i]
        # tup: ([3, 4], [2])
        for tup in tuples:
            ap, op = tup[0], tup[1]
            opt = [sents[i][j] for j in op]
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


# Prefer to order it and pick it up though because I do n't like the servers , one young woman in particular .####[([15], [11, 12, 13], 'NEG'), ([18, 19], [11, 12, 13], 'NEG')]

def get_annotated_aste_targets(sents, labels):
    annotated_targets = []
    num_sents = len(sents)
    for i in range(num_sents):
        tuples = labels[i]
        # tup: ([2], [5], 'NEG')
        for tup in tuples:
            ap, op, sent = tup[0], tup[1], tup[2]
            op = [sents[i][j] for j in op]
            # multiple OT for one AP
            if '[' in sents[i][ap[0]]:
                # print(i)
                if len(ap) == 1:
                    sents[i][ap[0]
                             ] = f"{sents[i][ap[0]][:-1]}, {' '.join(op)}]"
                else:
                    sents[i][ap[-1]
                             ] = f"{sents[i][ap[-1]][:-1]}, {' '.join(op)}]"
            else:
                annotation = f"{senttag2word[sent]}|{' '.join(op)}"
                if len(ap) == 1:
                    sents[i][ap[0]] = f"[{sents[i][ap[0]]}|{annotation}]"
                else:
                    sents[i][ap[0]] = f"[{sents[i][ap[0]]}"
                    sents[i][ap[-1]] = f"{sents[i][ap[-1]]}|{annotation}]"
        annotated_targets.append(sents[i])
    return annotated_targets


def get_annotated_tasd_targets(sents, labels):
    targets = []
    num_sents = len(sents)
    sents_str = [' '.join(s) for s in sents]
    for i in range(num_sents):
        s_str = sents_str[i]
        at_dict = {}
        for triplet in labels[i]:
            at, ac, polarity = triplet[0], triplet[1], triplet[2]
            if at in at_dict:
                at_dict[at][0].append(ac)
            else:
                at_dict[at] = [[ac], polarity]
        for at, ac_pol in at_dict.items():
            if len(ac_pol[0]) == 1:
                annotated_at = f"[{at}|{ac_pol[0][0]}|{ac_pol[1]}]"
            else:
                annotated_at = f"[{at}|{', '.join(ac_pol[0])}|{ac_pol[1]}]"
            if at != 'NULL':
                # print('at:', at, 'replaced_at:', annotated_at)
                s_str = s_str.replace(at, annotated_at)
            else:
                s_str += f" {annotated_at}"
        targets.append(s_str)
    return targets


def get_extraction_uabsa_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        if label == []:
            targets.append('None')
        else:
            all_tri = []
            for tri in label:
                if len(tri[0]) == 1:
                    a = sents[i][tri[0][0]]
                else:
                    start_idx, end_idx = tri[0][0], tri[0][-1]
                    a = ' '.join(sents[i][start_idx:end_idx+1])
                c = senttag2word[tri[1]]
                all_tri.append((a, c))
            label_strs = ['('+', '.join(label)+')' for label in all_tri]
            targets.append('; '.join(label_strs))
    return targets


def get_extraction_aope_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        all_tri = []
        for tri in label:
            if len(tri[0]) == 1:
                a = sents[i][tri[0][0]]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                a = ' '.join(sents[i][start_idx:end_idx+1])
            if len(tri[1]) == 1:
                b = sents[i][tri[1][0]]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                b = ' '.join(sents[i][start_idx:end_idx+1])
            all_tri.append((a, b))
        label_strs = ['('+', '.join(label)+')' for label in all_tri]
        targets.append('; '.join(label_strs))
    return targets


def get_extraction_tasd_targets(sents, labels):
    targets = []
    for label in labels:
        label_strs = ['('+', '.join(li)+')' for li in label]
        target = '; '.join(label_strs)
        targets.append(target)
    return targets


def get_extraction_aste_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        all_tri = []
        for tri in label:
            if len(tri[0]) == 1:
                a = sents[i][tri[0][0]]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                a = ' '.join(sents[i][start_idx:end_idx+1])
            if len(tri[1]) == 1:
                b = sents[i][tri[1][0]]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                b = ' '.join(sents[i][start_idx:end_idx+1])
            c = senttag2word[tri[2]]
            all_tri.append((a, b, c))
        label_strs = ['('+', '.join(lable)+')' for lable in all_tri]
        targets.append('; '.join(label_strs))
    return targets


def get_semantic_aste_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        all_tri = []
        for tri in label:
            if len(tri[0]) == 1:
                a = sents[i][tri[0][0]]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                a = ' '.join(sents[i][start_idx:end_idx+1])
            if len(tri[1]) == 1:
                b = sents[i][tri[1][0]]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                b = ' '.join(sents[i][start_idx:end_idx+1])
            c = senttag2word[tri[2]]
            all_tri.append(("Target:" + a, 'Opinion:' + b,  'Emotion:' + c))
        label_strs = [', '.join(lable) for lable in all_tri]
        targets.append('; '.join(label_strs))
    return targets


def get_prompt_aste_targets(sents, labels):
    # 需要处理一个 opinion 描述 多个 aspect 的情况
    # 需要标注aspect
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
    #         if '[' in sents[i][ap[0]]:
    #             pass
    #         else:
    #             if len(ap) == 1:
    #                 sents[i][ap[0]] = f"[{sents[i][ap[0]]}|aspect]"
    #             else:
    #                 sents[i][ap[0]] = f"[{sents[i][ap[0]]}"
    #                 sents[i][ap[-1]] = f"{sents[i][ap[-1]]}|aspect]"

    #         # annotation = f"opinion|aspect={' '.join(apStr)}|{senttag2word[sent]}"
    #         annotation = f"{senttag2word[sent]}|aspect={' '.join(apStr)}"
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

    annotated_targets = []
    num_sents = len(sents)
    for i in range(num_sents):
        tuples = labels[i]
        for tup in tuples:
            ap, op, sent = tup[0], tup[1], tup[2]
            opStr = [sents[i][j] for j in op]
            # multiple OT for one AP
            if '[' in sents[i][ap[0]]:
                # print(i)
                if len(ap) == 1:
                    sents[i][ap[0]
                             ] = f"{sents[i][ap[0]][:-1]}, {' '.join(opStr)}]"
                else:
                    sents[i][ap[-1]
                             ] = f"{sents[i][ap[-1]][:-1]}, {' '.join(opStr)}]"
            else:
                annotation = f"{senttag2word[sent]}|{' '.join(opStr)}"
                if len(ap) == 1:
                    sents[i][ap[0]] = f"[{sents[i][ap[0]]}|{annotation}]"
                else:
                    sents[i][ap[0]] = f"[{sents[i][ap[0]]}"
                    sents[i][ap[-1]] = f"{sents[i][ap[-1]]}|{annotation}]"

            if '[' in sents[i][op[0]]:
                # if len(op) == 1:
                #     sents[i][op[0]] = f"[{sents[i][op[0]][:-1]}, {' '.join(apStr)}]"
                # else:
                #     sents[i][op[-1]] = f"{sents[i][op[-1]][:-1]}, {' '.join(apStr)}]"
                pass
            else:
                if len(op) == 1:
                    sents[i][op[0]] = f"[{sents[i][op[0]]}|opinion]"
                else:
                    sents[i][op[0]] = f"[{sents[i][op[0]]}"
                    sents[i][op[-1]] = f"{sents[i][op[-1]]}|opinion]"

        annotated_targets.append(sents[i])
    return annotated_targets


def get_transformed_io(data_path, paradigm, task):
    """
    The main function to transform the Input & Output according to
    the specified paradigm and task
    """
    sents, labels = read_line_examples_from_file(data_path)

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]

    # Get target according to the paradigm
    # annotate the sents (with label info) as targets
    if paradigm == 'annotation':
        if task == 'uabsa':
            targets = get_annotated_uabsa_targets(sents, labels)
        elif task == 'aste':
            targets = get_annotated_aste_targets(sents, labels)
        elif task == 'tasd':
            targets = get_annotated_tasd_targets(sents, labels)
        elif task == 'aope':
            targets = get_annotated_aope_targets(sents, labels)
        else:
            raise NotImplementedError
    # directly treat label infor as the target
    elif paradigm == 'extraction':
        if task == 'uabsa':
            targets = get_extraction_uabsa_targets(sents, labels)
        elif task == 'aste':
            targets = get_extraction_aste_targets(sents, labels)
        elif task == 'tasd':
            targets = get_extraction_tasd_targets(sents, labels)
        elif task == 'aope':
            targets = get_extraction_aope_targets(sents, labels)
        else:
            raise NotImplementedError
    elif paradigm == 'semantic':
        if task == 'uabsa':
            # targets = get_extraction_uabsa_targets(sents, labels)
            pass
        elif task == 'aste':
            targets = get_semantic_aste_targets(sents, labels)
        elif task == 'tasd':
            # targets = get_extraction_tasd_targets(sents, labels)
            pass
        elif task == 'aope':
            # targets = get_extraction_aope_targets(sents, labels)
            pass
        else:
            raise NotImplementedError
    elif paradigm == 'prompt':
        if task == 'uabsa':
            # targets = get_extraction_uabsa_targets(sents, labels)
            pass
        elif task == 'aste':
            targets = get_prompt_aste_targets(sents, labels)
        elif task == 'tasd':
            # targets = get_extraction_tasd_targets(sents, labels)
            pass
        elif task == 'aope':
            # targets = get_extraction_aope_targets(sents, labels)
            pass
        else:
            raise NotImplementedError
    else:
        print('Unsupported paradigm!')
        raise NotImplementedError

    return inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, paradigm, task, max_len=128):
        # 'data/aste/rest16/train.txt'
        self.data_path = f'data/{task}/{data_dir}/{data_type}.txt'
        self.paradigm = paradigm
        self.task = task
        self.max_len = max_len
        self.tokenizer = tokenizer

        self.inputs = []
        self.targets = []
        # self.force_words_ids = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        # force_words_ids = self.force_words_ids
        # might need to squeeze
        src_mask = self.inputs[index]["attention_mask"].squeeze()
        # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask,
                # 'force_words_ids': force_words_ids
                }

    def _build_examples(self):

        inputs, targets = get_transformed_io(
            self.data_path, self.paradigm, self.task)
        # constraint_str = ['[', ']', '|','opinion','aspect','positive','neutral','negative',]
        for i in range(len(inputs)):
            # 为了获得 标注结果对应的文本，所以提前对句子进行了分词 这里就需要拼接回原句
            input = ' '.join(inputs[i])
            if self.paradigm == 'annotation':
                if self.task != 'tasd':
                    target = ' '.join(targets[i])
                else:
                    target = targets[i]
            elif self.paradigm == 'prompt':
                if self.task != 'tasd':
                    target = ' '.join(targets[i])
            else:
                target = targets[i]

            tokenized_input = self.tokenizer(
                input,   padding='max_length', max_length=128, truncation=True, return_tensors="pt")
            tokenized_target = self.tokenizer(
                target,  padding='max_length', max_length=128, truncation=True, return_tensors="pt")
            
            # force_words_ids = self.tokenizer(constraint_str + inputs[i]).input_ids
            # force_words_ids = self.tokenizer(inputs[i]).input_ids
            
            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
            # self.force_words_ids.append(force_words_ids)


def write_results_to_log(log_file_path, best_test_result, args, dev_results, test_results, global_steps):
    """
    Record dev and test results to log file
    """
    local_time = time.asctime(time.localtime(time.time()))
    exp_settings = "Exp setting: {0} on {1} under {2} | {3:.4f} | ".format(
        args.task, args.dataset, args.paradigm, best_test_result
    )
    train_settings = "Train setting: bs={0}, lr={1}, num_epochs={2}".format(
        args.train_batch_size, args.learning_rate, args.num_train_epochs
    )
    results_str = "\n* Results *:  Dev  /  Test  \n"

    metric_names = ['f1', 'precision', 'recall']
    for gstep in global_steps:
        results_str += f"Step-{gstep}:\n"
        for name in metric_names:
            name_step = f'{name}_{gstep}'
            results_str += f"{name:<8}: {dev_results[name_step]:.4f} / {test_results[name_step]:.4f}"
            results_str += ' '*5
        results_str += '\n'

    log_str = f"{local_time}\n{exp_settings}\n{train_settings}\n{results_str}\n\n"

    with open(log_file_path, "a+") as f:
        f.write(log_str)
