import argparse
import os
import logging
import time
# import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from data_utils import ABSADataset, data_samples
from data_utils import write_results_to_log, read_line_examples_from_file
from eval_utils import compute_scores
from models.t5FineTuner import T5FineTuner, Tokenizer
# from models.BART_FineTuner import BARTFineTuner, Tokenizer
# from models.DeBERTa_Generator import DeBERTaGenerator, Tokenizer

logger = logging.getLogger(__name__)

def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument(
        "--task",
        default='aste',
        type=str,
        required=True,
        help="The name of the task, selected from: [uabsa, aste, tasd, aope]")
    parser.add_argument(
        "--dataset",
        default='rest16',
        type=str,
        required=True,
        help="The name of the dataset, selected from: [laptop14, rest14, rest15, rest16]"
    )
    parser.add_argument("--model_name_or_path",
                        default='t5-base',
                        type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument(
        "--paradigm",
        default='semantic',
        type=str,
        required=True,
        help="The way to construct target sentence, selected from: [annotation, extraction, semantic, prompt]"
    )
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval",
                        action='store_true',
                        help="Whether to run direct eval on the dev/test set.")

    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs",
                        default=20,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    args = parser.parse_args()

    # set up output dir which looks like './aste/rest14/extraction/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    task_dir = f"./outputs/{args.task}"
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    task_dataset_dir = f"{task_dir}/{args.dataset}"
    if not os.path.exists(task_dataset_dir):
        os.mkdir(task_dataset_dir)

    output_dir = f"{task_dataset_dir}/{args.paradigm}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.args.output_dir,
                                                "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def evaluate(data_loader, model, paradigm, task, sents):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device(
        f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.model.to(device)

    model.model.eval()
    outputs, targets = [], []
    for batch in tqdm(data_loader):
        # need to push the data to device
        outs = model.model.generate(
            input_ids=batch['source_ids'].to(device),
            attention_mask=batch['source_mask'].to(device),
            max_length=128)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
        ]

        outputs.extend(dec)
        targets.extend(target)
        
    log_file_path = f"results_log/{task}-outputs.txt"
    with open(log_file_path, "a+", encoding='utf-8') as f:
        f.truncate()
        for i  in range(len(outputs)) :
            if targets[i] != outputs[i]:
                f.write(targets[i] + '\n')
                f.write(outputs[i] + '\n')
                f.write('\n')
            
    raw_scores, fixed_scores, all_labels, all_preds, all_preds_fixed = compute_scores(
        outputs, targets, sents, paradigm, task)
    # results = {
    #     'raw_scores': raw_scores,
    #     'fixed_scores': fixed_scores,
    #     'labels': all_labels,
    #     'preds': all_preds,
    #     'preds_fixed': all_preds_fixed
    # }
    # pickle.dump(
    #     results,
    #     open(
    #         f"{args.output_dir}/results-{args.task}-{args.dataset}-{args.paradigm}.pickle",
    #         'wb'))

    return raw_scores, fixed_scores


if __name__ == '__main__':
    # initialization
    args = init_args()
    print("\n", "=" * 30, f"NEW EXP: {args.task.upper()} on {args.dataset}",
          "=" * 30, "\n")

    # Instead of `torch.manual_seed(...) 设置随机数种子，方便下次复现实验结果，每次执行文件，获得的随机数是相同的
    seed_everything(args.seed)

    tokenizer = Tokenizer()

    data_samples(args, tokenizer, 'dev')

    # training process
    if args.do_train:
        print("\n****** Conduct Training ******")
        model = T5FineTuner(args)

        # filepath  prefix="ckt",
        # enable_checkpointing = pl.callbacks.ModelCheckpoint(
        #     dirpath=args.output_dir, monitor='val_loss', mode='min', save_top_k=3)

        # prepare for trainer
        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            # gpus=1,
            devices=1,
            accelerator="gpu",
            # auto_select_gpus=True,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            # enable_checkpointing=enable_checkpointing,
            # callbacks=[LoggingCallback()],
        )

        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

        # save the final model
        # model.model.save_pretrained(args.output_dir)

        print("Finish training and saving the model!")

    if args.do_eval:

        print("\n****** Conduct Evaluating ******")

        # model = T5FineTuner(args)
        dev_results, test_results = {}, {}
        best_f1, best_checkpoint, best_epoch = -999999.0, None, None
        all_checkpoints, all_epochs = [], []

        # retrieve all the saved checkpoints for model selection
        saved_model_dir = args.output_dir
        for f in os.listdir(saved_model_dir):
            file_name = os.path.join(saved_model_dir, f)
            if 'cktepoch' in file_name:
                all_checkpoints.append(file_name)

        # conduct some selection (or not)
        print(
            f"We will perform validation on the following checkpoints: {all_checkpoints}"
        )

        # load dev and test datasets
        dev_dataset = ABSADataset(tokenizer,
                                  data_dir=args.dataset,
                                  data_type='dev',
                                  paradigm=args.paradigm,
                                  task=args.task,
                                  max_len=args.max_seq_length)
        dev_loader = DataLoader(
            dev_dataset, batch_size=32, num_workers=3)

        test_dataset = ABSADataset(tokenizer,
                                   data_dir=args.dataset,
                                   data_type='test',
                                   paradigm=args.paradigm,
                                   task=args.task,
                                   max_len=args.max_seq_length)
        test_loader = DataLoader(
            test_dataset, batch_size=32, num_workers=3)

        for checkpoint in all_checkpoints:
            epoch = checkpoint.split(
                '=')[-1][:-5] if len(checkpoint) > 1 else ""
            # only perform evaluation at the specific epochs ("15-19")
            # eval_begin, eval_end = args.eval_begin_end.split('-')
            if 0 <= int(epoch) < 100:
                all_epochs.append(epoch)

                # reload the model and conduct inference
                print(f"\nLoad the trained model from {checkpoint}...")
                model_ckpt = torch.load(checkpoint)
                model = T5FineTuner(model_ckpt['hyper_parameters'])
                model.load_state_dict(model_ckpt['state_dict'])

                dev_result = evaluate(
                    dev_loader, model, args.paradigm, args.task)
                if dev_result['f1'] > best_f1:
                    best_f1 = dev_result['f1']
                    best_checkpoint = checkpoint
                    best_epoch = epoch

                # add the global step to the name of these metrics for recording
                # 'f1' --> 'f1_1000'
                dev_result = dict(
                    (k + '_{}'.format(epoch), v) for k, v in dev_result.items())
                dev_results.update(dev_result)

                test_result = evaluate(test_loader, model, args.paradigm,
                                       args.task)
                test_result = dict(
                    (k + '_{}'.format(epoch), v) for k, v in test_result.items())
                test_results.update(test_result)

        # print test results over last few steps
        print(f"\n\nThe best checkpoint is {best_checkpoint}")
        best_step_metric = f"f1_{best_epoch}"
        print(f"F1 scores on test set: {test_results[best_step_metric]:.4f}")

        print("\n* Results *:  Dev  /  Test  \n")
        metric_names = ['f1', 'precision', 'recall']
        for epoch in all_epochs:
            print(f"Epoch-{epoch}:")
            for name in metric_names:
                name_step = f'{name}_{epoch}'
                print(
                    f"{name:<10}: {dev_results[name_step]:.4f} / {test_results[name_step]:.4f}",
                    sep='  ')
            print()

        results_log_dir = './results_log'
        if not os.path.exists(results_log_dir):
            os.mkdir(results_log_dir)
        log_file_path = f"{results_log_dir}/{args.task}-{args.dataset}.txt"
        write_results_to_log(log_file_path, test_results[best_step_metric], args,
                             dev_results, test_results, all_epochs)

    # evaluation process
    if args.do_direct_eval:
        print("\n****** Conduct Evaluating with the last state ******")

        # model = T5FineTuner(args)

        # print("Reload the model")
        # model.model.from_pretrained(args.output_dir)

        sents, _ = read_line_examples_from_file(
            f'data/{args.task}/{args.dataset}/test.txt')

        test_dataset = ABSADataset(tokenizer,
                                   data_dir=args.dataset,
                                   data_type='test',
                                   paradigm=args.paradigm,
                                   task=args.task,
                                   max_len=args.max_seq_length)
        test_loader = DataLoader(
            test_dataset, batch_size=32, num_workers=3)
        # print(test_loader.device)
        raw_scores, fixed_scores = evaluate(test_loader, model, args.paradigm,
                                            args.task, sents)
        # print(scores)

        # write to file
        log_file_dir = "results_log"
        if not os.path.exists(log_file_dir):
            os.mkdir(log_file_dir)
        log_file_path = f"results_log/{args.task}-{args.dataset}.txt"
        local_time = time.asctime(time.localtime(time.time()))
        exp_settings = f"{args.task} on {args.dataset} under {args.paradigm}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}"
        # exp_results = f"Raw F1 = {raw_scores['f1']:.4f}, Fixed F1 = {fixed_scores['f1']:.4f}"
        exp_results = f"Raw   F1 = {raw_scores['f1']:.4f}, precision = {raw_scores['precision']:.4f}, recall = {raw_scores['recall']:.4f}\n"
        exp_results += f"Fixed F1 = {fixed_scores['f1']:.4f}, precision = {fixed_scores['precision']:.4f}, recall = {fixed_scores['recall']:.4f} "
        log_str = '============================================================\n'
        log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"
        with open(log_file_path, "a+") as f:
            f.write(log_str)
