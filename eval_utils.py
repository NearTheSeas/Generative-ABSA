# This file contains the evaluation functions

import re
import editdistance
import time

import spacy

nlp = spacy.load("en_core_web_md")

sentiment_word_list = ['positive', 'negative', 'neutral']
aspect_cate_list = ['location general',
                    'food prices',
                    'food quality',
                    'ambience general',
                    'service general',
                    'restaurant prices',
                    'drinks prices',
                    'restaurant miscellaneous',
                    'drinks quality',
                    'drinks style_options',
                    'restaurant general',
                    'food style_options']

err_split = '---'


def extract_spans_semantic(task, seq):
    extractions = []
    if task == 'uabsa' and seq.lower() == 'none':
        return []
    else:
        if task in ['uabsa', 'aope']:
            all_pt = seq.split('; ')
            for pt in all_pt:
                pt = pt[1:-1]
                try:
                    a, b = pt.split(', ')
                except ValueError:
                    a, b = '', ''
                extractions.append((a, b))
        elif task in ['tasd', 'aste']:
            #  Target:food, Opinion:cheap, Sentiment:positive; Target:waiters, Opinion:nice, Sentiment:positive
            all_pt = seq.split('; ')
            for pt in all_pt:
                try:
                    arr = [item.split(':') for item in pt.split(', ')]
                    a, b, c = [item[1] if len(
                        item) > 1 else '' for item in arr]
                except ValueError:
                    a, b, c = '', '', ''
                extractions.append((a, b, c))
        return extractions

def extract_spans_prompt(task, seq):
    extractions = []
    if task == 'uabsa' and seq.lower() == 'none':
        return []
    else:
        if task in ['uabsa', 'aope']:
            all_pt = seq.split('; ')
            for pt in all_pt:
                pt = pt[1:-1]
                try:
                    a, b = pt.split(', ')
                except ValueError:
                    a, b = '', ''
                extractions.append((a, b))
        elif task in ['tasd', 'aste']:
            extractions = extract_triplets_prompt(seq)
        return extractions

def extract_spans_extraction(task, seq):
    extractions = []
    if task == 'uabsa' and seq.lower() == 'none':
        return []
    else:
        if task in ['uabsa', 'aope']:
            all_pt = seq.split('; ')
            for pt in all_pt:
                pt = pt[1:-1]
                try:
                    a, b = pt.split(', ')
                except ValueError:
                    a, b = '', ''
                extractions.append((a, b))
        elif task in ['tasd', 'aste']:
            all_pt = seq.split('; ')
            for pt in all_pt:
                pt = pt[1:-1]
                try:
                    a, b, c = pt.split(', ')
                except ValueError:
                    a, b, c = '', '', ''
                extractions.append((a, b, c))
        return extractions


def extract_spans_annotation(task, seq):
    if task in ['aste', 'tasd']:
        extracted_spans = extract_triplets(seq)
    elif task in ['aope', 'uabsa']:
        extracted_spans = extract_pairs(seq)

    return extracted_spans


def extract_pairs(seq):
    aps = re.findall('\[.*?\]', seq)
    aps = [ap[1:-1] for ap in aps]
    pairs = []
    for ap in aps:
        # the original sentence might have
        try:
            at, ots = ap.split('|')
        except ValueError:
            at, ots = '', ''

        if ',' in ots:     # multiple ots
            for ot in ots.split(', '):
                pairs.append((at, ot))
        else:
            pairs.append((at, ots))
    return pairs


def extract_triplets(seq):
    aps = re.findall('\[.*?\]', seq)
    aps = [ap[1:-1] for ap in aps]
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

def extract_triplets_prompt(seq):
    # ops = re.findall('\[.*?\]', seq)
    # ops = list(filter(lambda x:len(x.split('|')) ==  3 , ops))
    # ops = [ap[1:-1] for ap in ops]
    # triplets = []
    # for op in ops:
    #     try:
    #         a, b, c = op.split('|')
    #     except ValueError:
    #         a, b, c = '', '', ''
    #     # for ASTE
    #     if b in sentiment_word_list:
    #         if '=' in c:
    #             aspects = c.split('=')[1]
    #             aspects = aspects.split(', ')
    #             for item in aspects:
    #                 triplets.append((a, b, item))
    #     # for TASD
    #     else:
    #         if ',' in b:
    #             for ac in b.split(', '):
    #                 triplets.append((a, ac, c))
    #         else:
    #             triplets.append((a, b, c))
    
    aps = re.findall('\[.*?\]', seq)
    aps = list(filter(lambda x:len(x.split('|')) ==  3 , aps))
    aps = [ap[1:-1] for ap in aps]
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
 

def recover_terms_with_editdistance(original_term, sent):
    words = original_term.split(' ')
    new_words = []
    for word in words:
        edit_dis = []
        for token in sent:
            edit_dis.append(editdistance.eval(word, token))
        smallest_idx = edit_dis.index(min(edit_dis))
        new_words.append(sent[smallest_idx])
    new_term = ' '.join(new_words)
    return new_term

# 性能不行
def recover_terms_with_editdistance2(original_term, sent):
    # french_fries.similarity(burgers)
    words = original_term.split(' ')
    new_words = []
    for word in words:
        edit_dis = []
        nlpword = nlp(word)
        nlpsent = nlp(' '.join(sent))
        for token in nlpsent:
            # token = nlp(token)
            # edit_dis.append(editdistance.eval(word, token))
            if (token and token.vector_norm):
                edit_dis.append(nlpword.similarity(token))
            else:
                edit_dis.append(-1)
        # smallest_idx = edit_dis.index(min(edit_dis))
        smallest_idx = edit_dis.index(max(edit_dis))
        new_words.append(sent[smallest_idx])
    new_term = ' '.join(new_words)
    return new_term


def fix_preds_uabsa(all_pairs, sents):

    all_new_pairs = []
    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                # AT not in the original sentence
                if pair[0] not in ' '.join(sents[i]):
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(pair[0], sents[i])
                else:
                    new_at = pair[0]

                if pair[1] not in sentiment_word_list:
                    new_sentiment = recover_terms_with_editdistance(
                        pair[1], sentiment_word_list)
                else:
                    new_sentiment = pair[1]

                new_pairs.append((new_at, new_sentiment))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_preds_aope(all_pairs, sents):

    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                # print(pair)
                # AT not in the original sentence
                if pair[0] not in ' '.join(sents[i]):
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(pair[0], sents[i])
                else:
                    new_at = pair[0]

                # OT not in the original sentence
                ots = pair[1].split(', ')
                new_ot_list = []
                for ot in ots:
                    if ot not in ' '.join(sents[i]):
                        # print('Issue')
                        new_ot_list.append(
                            recover_terms_with_editdistance(ot, sents[i]))
                    else:
                        new_ot_list.append(ot)
                new_ot = ', '.join(new_ot_list)

                new_pairs.append((new_at, new_ot))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


# for ASTE
def fix_preds_aste(all_pairs, sents):

    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                # two formats have different orders
                p0, p1, p2 = pair
                # for annotation-type
                if p1 in sentiment_word_list:
                    at, ott, ac = p0, p2, p1
                    io_format = 'annotation'
                # for extraction type
                elif p2 in sentiment_word_list:
                    at, ott, ac = p0, p1, p2
                    io_format = 'extraction'

                # print(pair)
                # AT not in the original sentence
                if at not in ' '.join(sents[i]):
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(at, sents[i])
                else:
                    new_at = at

                if ac not in sentiment_word_list:
                    new_sentiment = recover_terms_with_editdistance(
                        ac, sentiment_word_list)
                else:
                    new_sentiment = ac

                # OT not in the original sentence
                ots = ott.split(', ')
                new_ot_list = []
                for ot in ots:
                    if ot not in ' '.join(sents[i]):
                        # print('Issue')
                        new_ot_list.append(
                            recover_terms_with_editdistance(ot, sents[i]))
                    else:
                        new_ot_list.append(ot)
                new_ot = ', '.join(new_ot_list)
                if io_format == 'extraction':
                    new_pairs.append((new_at, new_ot, new_sentiment))
                else:
                    new_pairs.append((new_at, new_sentiment, new_ot))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            log_file_path = f"results_log/edit-distance-outputs.txt"

            with open(log_file_path, "a+", encoding='utf-8') as f:
                f.write(';'.join(','.join(item) for item in pairs))
                f.write('---')
                f.write(';'.join(','.join(item) for item in new_pairs))
                f.write('\n')
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_preds_tasd(all_pairs, sents):

    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                # print(pair)
                # AT not in the original sentence
                sents_and_null = ' '.join(sents[i]) + 'NULL'
                if pair[0] not in sents_and_null:
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(pair[0], sents[i])
                else:
                    new_at = pair[0]

                # AC not in the list
                acs = pair[1].split(', ')
                new_ac_list = []
                for ac in acs:
                    if ac not in aspect_cate_list:
                        new_ac_list.append(
                            recover_terms_with_editdistance(ac, aspect_cate_list))
                    else:
                        new_ac_list.append(ac)
                new_ac = ', '.join(new_ac_list)

                if pair[2] not in sentiment_word_list:
                    new_sentiment = recover_terms_with_editdistance(
                        pair[2], sentiment_word_list)
                else:
                    new_sentiment = pair[2]

                new_pairs.append((new_at, new_ac, new_sentiment))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_pred_with_editdistance(all_predictions, sents, task):
    if task == 'uabsa':
        fixed_preds = fix_preds_uabsa(all_predictions, sents)
    elif task == 'aope':
        fixed_preds = fix_preds_aope(all_predictions, sents)
    elif task == 'aste':
        # fixed_preds = fix_preds_aste(all_predictions, sents)
        fixed_preds = fix_preds_aste(all_predictions, sents)
    elif task == 'tasd':
        fixed_preds = fix_preds_tasd(all_predictions, sents)
    else:
        print("*** Unimplemented Error ***")
        fixed_preds = all_predictions

    return fixed_preds


def compute_f1_scores(pred_pt, gold_pt,io_format, task):
    """
    Function to compute F1 scores with pred and gold pairs/triplets
    The input needs to be already processed
    """
    # number of true postive, gold standard, predicted aspect terms
    n_tp, n_gold, n_pred = 0, 0, 0
    errorList = []

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])
        mis = 0

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1
            else:
                mis += 1
                # print( pred_pt[i], '-----------', gold_pt[i])
                # errorList.append(t)
                # print(t, gold_pt[i])
        if mis:
            errorList.append(';'.join([','.join(list(item)) for item in pred_pt[i]]) +
                             err_split + ';'.join([','.join(list(item)) for item in gold_pt[i]]))
    log_error(errorList, io_format, task)

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def compute_scores(pred_seqs, gold_seqs, sents, io_format, task):
    """
    compute metrics for multiple tasks
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_predictions = [], []

    for i in range(num_samples):
        if io_format == 'annotation':
            gold_list = extract_spans_annotation(task, gold_seqs[i])
            pred_list = extract_spans_annotation(task, pred_seqs[i])
        elif io_format == 'extraction':
            gold_list = extract_spans_extraction(task, gold_seqs[i])
            pred_list = extract_spans_extraction(task, pred_seqs[i])
        elif io_format == 'semantic':
            gold_list = extract_spans_semantic(task, gold_seqs[i])
            pred_list = extract_spans_semantic(task, pred_seqs[i])
        elif io_format == 'prompt':
            gold_list = extract_spans_prompt(task, gold_seqs[i])
            pred_list = extract_spans_prompt(task, pred_seqs[i])
        all_labels.append(gold_list)
        all_predictions.append(pred_list)

    print("\nResults of raw output")
    raw_scores = compute_f1_scores(all_predictions, all_labels, io_format,task)
    print(raw_scores)

    # fix the issues due to generation
    all_predictions_fixed = fix_pred_with_editdistance(
        all_predictions, sents, task)
    print("\nResults of fixed output")
    fixed_scores = compute_f1_scores(all_predictions_fixed, all_labels,io_format, task)
    print(fixed_scores)

    return raw_scores, fixed_scores, all_labels, all_predictions, all_predictions_fixed


def log_error(content,io_format, task):
    # local_time = time.asctime(time.localtime(time.time()))
    local_time = time.gmtime()
    local_time = time.strftime("%Y%m%d_%H_%M",local_time)
    log_file_path = f"results_log/{task}-{io_format}-errors-{local_time}.txt"
    with open(log_file_path, "a+", encoding='utf-8') as f:
        for item in content:
            a, b = item.split(err_split)
            f.write(a + '\n')
            f.write(b+'\n\n')
