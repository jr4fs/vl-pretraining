# coding=utf-8
# Copyleft 2019 project LXRT.
import neptune.new as neptune

import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np
from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
import json 
import os

def get_data_tuple(splits: str, subset: str, bs:int, shuffle=False, drop_last=False, sampling_ids=None) -> DataTuple:
    dset = VQADataset(splits, subset, sampling_ids)
    if splits != 'minival':
        index_dset = len(dset.data) % bs
        if index_dset != 0:
            dset.data = dset.data[:-index_dset] 
    tset = VQATorchDataset(dset)
    if splits != 'minival':
        index_tset = len(tset.data) % bs
        if index_tset != 0:
            tset.data = tset.data[:-index_tset]
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )
    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

def params_to_sampling_ids(params):
    # return path to sampling ids 
    if type(params) == str:
        return params
    else:
        print('implement parsing') 
    # else -- parse from params
    # params = {'sampling_method': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
    #         'budget': trial.suggest_int('max_depth', 1, 30),
    #         'alpha': trial.suggest_int('num_leaves', 2, 100),
    #         'beta': trial.suggest_int('min_data_in_leaf', 10, 1000),
    #         'norm': trial.suggest_uniform('feature_fraction', 0.1, 1.0)}

class VQA:
    def __init__(self, sampling_ids=None):
        if sampling_ids != None:
            sampling_ids_path = params_to_sampling_ids(sampling_ids)
        else:
            sampling_ids_path = None
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, args.subset, bs=args.batch_size, shuffle=True, drop_last=False, sampling_ids=sampling_ids_path
        )

        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, args.subset, bs=1024,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        if args.subset != 'None':
            self.loss_fxn = nn.CrossEntropyLoss()
        else:
            self.loss_fxn = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple, run=None):

        dset, loader, evaluator = train_tuple
        run["# Examples"] = len(loader.dataset)
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        all_loss = []
        valid_scores = []
        train_scores = []
        training_stats = []
        for epoch in range(args.epochs):
            quesid2ans = {}

            for i, (ques_id, feats, boxes, sent, target, img_id) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                softmax = torch.nn.Softmax(dim=1)
                logit_softmax = softmax(logit)
                gt_preds_probability_softmax= torch.squeeze(logit_softmax.gather(1, torch.unsqueeze(target, 1)))

                #assert logit.dim() == target.dim() == 2
                loss = self.loss_fxn(logit, target)
                loss = loss * logit.size(1)
                all_loss.append(loss.detach().cpu().numpy())


                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                if args.subset != None:
                    score, label = logit_softmax.max(1)
                else:
                    score, label = logit.max(1)
                

                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans


                for idx, question in enumerate(sent):
                    preds = dset.label2ans[np.squeeze(label.cpu().numpy()[idx].astype(int))]
                    ans_gt = dset.label2ans[np.squeeze(target.cpu().numpy()[idx].astype(int))]
                    if args.subset != None:
                        training_stats.append({
                            "Epoch": int(epoch),
                            "Question ID": int(ques_id[idx]),
                            "Image ID": str(img_id[idx]),
                            "Question": str(question),
                            "Target": str(ans_gt),
                            "Prediction": str(preds),
                            "GT Probability": float(gt_preds_probability_softmax[idx])
                            }
                    )
                    
                if i%1000 ==0:
                    for idx, question in enumerate(sent):
                        ans_gt = dset.label2ans[target.cpu().numpy()[idx]]
                        preds = dset.label2ans[label.cpu().numpy()[idx]]
                        preds_str = "Image ID: " + img_id[idx] + "\n Question: " + question + "\n ans_gt: " + ans_gt + "\n preds: " + preds + "\n"
                        with open(self.output + "/log_preds.log", 'a') as preds_file:
                            preds_file.write(preds_str)
                            preds_file.flush()  

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)
            train_scores.append(evaluator.evaluate(quesid2ans) * 100.)
            if run != None:
                run["train acc"].log(evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                valid_scores.append(valid_score)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)
                if run != None:
                    run["val acc"].log(valid_score * 100.)
                    run["best acc"].log(best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        with open(self.output+'/datamaps_stats.json', 'w') as json_file:
            json.dump(training_stats, json_file, 
                                indent=4,  
                                separators=(',',': '))
        self.save("LAST")
        return best_valid

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                if args.subset != None:
                    softmax = torch.nn.Softmax()
                    score, label = softmax(logit).max(1)
                else:
                    score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target, img_id) in enumerate(loader):
            if args.subset != None:
                label = target
            else:
                _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)

def run_training(run):
    # Neptune logging
    if args.sampling_ids != None:
        run["sampling_ids"] = os.path.basename(args.sampling_ids)
        run["sampling_method"] = args.sampling_method
        run["sampling_model"] = args.sampling_model
        run["training_budget"] = args.training_budget
        if args.sampling_method == 'beta':
            run["alpha"] = args.alpha
            run["beta"] = args.beta
            run["norm"] = args.norm
        else:
            run["alpha"] = '-'
            run["beta"] = '-'
            run["norm"] = '-'
    else:
        run["sampling_ids"] = '-'
        run["sampling_method"] = '-'
        run["sampling_model"] = '-'
        run["training_budget"] = 100
        run["alpha"] = '-'
        run["beta"] = '-'
        run["norm"] = '-'
    if args.subset!= None:
        run["subset"] = args.subset
    else:
        run["subset"] = '-'
        run["training_budget"] = 'all vqa'
    run["training_run"] = args.output
    run["learning_rate"] = args.lr
    run["optimizer"] = args.optim

    # Build Class
    vqa = VQA(args.sampling_ids)

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, args.subset, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('minival', args.subset, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'minival_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', vqa.train_tuple.dataset.splits)
        if vqa.valid_tuple is not None:
            print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple,run)

def neptune_monitor(study, trial):
    # log hyperparameters for neptune
    return 

def objective(trial):
    params = {'sampling_method': trial.suggest_categorical("sampling_method", ["beta", "random"]),
              'budget': trial.suggest_categorical("training_budget", [10, 20, 30]),
              'alpha': trial.suggest_categorical("alpha", [1, 2]),
              'beta': trial.suggest_categorical("beta", [1, 2]),
              'norm': trial.suggest_categorical("norm", ['pvals', 'var_counts', 'gaussian_kde', 'cosine', 'epanechnikov', 'exponential', 'gaussian', 'linear', 'tophat'])}
    # define using only sampling method, and budget for random sampling and the rest for beta sampling 
    
    # Neptune logging
    # run["sampling_ids"] = os.path.basename(args.sampling_ids)
    # run["sampling_method"] = args.sampling_method
    # run["sampling_model"] = args.sampling_model
    # run["training_budget"] = args.training_budget
    # if args.sampling_method == 'beta':
    #     run["alpha"] = args.alpha
    #     run["beta"] = args.beta
    #     run["norm"] = args.norm
    # else:
    #     run["alpha"] = '-'
    #     run["beta"] = '-'
    #     run["norm"] = '-'
    #     run["subset"] = args.subset
    # run["training_run"] = args.output
    # run["learning_rate"] = args.lr
    # run["optimizer"] = args.optim

    # Build Class
    vqa = VQA(params)
    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)
    print('Splits in Train data:', vqa.train_tuple.dataset.splits)
    if vqa.valid_tuple is not None:
        print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
        print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
    else:
        print("DO NOT USE VALIDATION")
    valid_score = vqa.train(vqa.train_tuple, vqa.valid_tuple)
    return valid_score


if __name__ == "__main__":
    if args.optuna == False:
        run = neptune.init(
            project="jranjit/vqa-training-data-selection",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOTJlZTEzNS00M2M1LTQwODMtYWQ3OS0zYTMxZGY3NTYwMjIifQ==",
        )  # your credentials
        run_training(run)
        run.stop()
    else:
        print("here")


