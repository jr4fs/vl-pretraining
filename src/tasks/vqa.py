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
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator
from os import path

# run = neptune.init(
#     project="vqa-training/vqa-training",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOTJlZTEzNS00M2M1LTQwODMtYWQ3OS0zYTMxZGY3NTYwMjIifQ==",
# )  # your credentials

# run = neptune.init(
#     project="vqa-training/vqa-training-myo",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOTJlZTEzNS00M2M1LTQwODMtYWQ3OS0zYTMxZGY3NTYwMjIifQ==",
# )  # your credentials

run = neptune.init_run(project='vqa-training/vqa-training-myo', 
                       api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOTJlZTEzNS00M2M1LTQwODMtYWQ3OS0zYTMxZGY3NTYwMjIifQ=="
)

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

def get_gqa_tuple(splits: str, subset: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(splits, subset)
    tset = GQATorchDataset(dset)
    evaluator = GQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

class VQA:
    def __init__(self, sampling_ids=None):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, args.subset, bs=args.batch_size, shuffle=True, drop_last=False, sampling_ids=sampling_ids
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
        if args.multiclass == True:
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

    def train(self, train_tuple, eval_tuple):

        dset, loader, evaluator = train_tuple
        run["# Examples"] = len(loader.dataset)
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        all_loss = []
        valid_scores = []
        train_scores = []
        # training_stats = []
        for epoch in range(args.epochs):
            training_stats = []
            quesid2ans = {}

            for i, (ques_id, feats, boxes, sent, target, img_id) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)

                #assert logit.dim() == target.dim() == 2
                loss = self.loss_fxn(logit, target)
                loss = loss * logit.size(1)
                all_loss.append(loss.detach().cpu().numpy())


                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                if args.multiclass == True:
                    softmax = torch.nn.Softmax(dim=1)
                    logit_softmax = softmax(logit)
                    gt_preds_probability_softmax = torch.squeeze(logit_softmax.gather(1, torch.unsqueeze(target, 1))) # Batchwise
                    #score, label = logit_softmax.max(1)
                    score, label = logit.max(1)
                else:
                    sigmoid = torch.nn.Sigmoid()
                    logit_sigmoid = sigmoid(logit)
                    score, label = logit.max(1) # gets the max predicted label for each instance 
                    target_bool = (target>0).long()
                    gt_preds_probability_sigmoid = logit_sigmoid * target_bool 

                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

                if args.multiclass == True:
                    for idx, question in enumerate(sent):
                        preds = dset.label2ans[np.squeeze(label.cpu().numpy()[idx].astype(int))]
                        ans_gt = dset.label2ans[np.squeeze(target.cpu().numpy()[idx].astype(int))]
                        
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
                else:
                    for idx, question in enumerate(sent):
                        preds = dset.label2ans[np.squeeze(label.cpu().numpy()[idx].astype(int))]
                        target_numpy = target_bool.cpu().numpy()[idx]
                        targets_indices = np.nonzero(target_numpy) # get indices of groundtruth 
                        target_indices_list = []
                        for i in targets_indices[0]:
                            target_indices_list.append(i)
                        #print(target_indices_list)

                        all_ans_gt = []
                        all_probs = []
                        for target_idx in target_indices_list:
                            #print("target idx: ", target_idx)
                            all_ans_gt.append(dset.label2ans[target_idx])
                        
                        
                        all_probs_sigmoid = logit_sigmoid.detach().cpu().numpy()[idx]
                        top_ans = []
                        all_probs_sigmoid_list = []
                        top_idx = np.argpartition(all_probs_sigmoid, -10)[-10:]
                        top_ten_probs = all_probs_sigmoid[top_idx]
                        for prob_sigmoid in top_ten_probs:
                            all_probs_sigmoid_list.append(str(prob_sigmoid))
                        for top in top_idx:
                            top_ans.append(dset.label2ans[top])

                        
                        probs_sigmoid = gt_preds_probability_sigmoid.detach().cpu().numpy()[idx]
                        probs = probs_sigmoid[np.nonzero(probs_sigmoid)] # get values at the ground truth targets
                        for x in probs:
                            all_probs.append(str(x))

                        datum = dset.id2datum[ques_id[idx].item()]
                        answer_type = datum['answer_type']
                        question_type = datum['question_type']
                        label_scores = datum['label']


                        training_stats.append({
                            "Epoch": int(epoch),
                            "Question ID": int(ques_id[idx]),
                            "Image ID": str(img_id[idx]),
                            "Question": str(question),
                            "Target": ', '.join(all_ans_gt),
                            "Prediction": str(preds),
                            "GT Probability": ', '.join(all_probs),
                            "Top Probabilities": ', '.join(all_probs_sigmoid_list),
                            "Top Answers": ', '.join(top_ans),
                            "Answer Type": str(answer_type),
                            "Question Type": str(question_type),
                            "Label": label_scores
                            }
                    )

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)
            train_scores.append(evaluator.evaluate(quesid2ans) * 100.)
            run["train acc"].log(evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                valid_scores.append(valid_score)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)
                run["val acc"].log(valid_score * 100.)
                run["best acc"].log(best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()
            self.save("Epoch"+str(epoch))


            # save training dymamics 
            filename = self.output+'/datamaps_stats.json'
            prev_dynamics = []
            if path.isfile(filename) is False:
                with open(filename, 'w') as json_file:
                    json.dump(training_stats, json_file, 
                                        indent=4,  
                                        separators=(',',': '))
            else:
                with open(filename) as fp:
                    prev_dynamics = json.load(fp)
                prev_dynamics.extend(training_stats)
                with open(filename, 'w') as json_file:
                    json.dump(prev_dynamics, json_file, 
                                        indent=4,  
                                        separators=(',',': '))

        self.save("LAST")
        #return best_valid * 100.

    def predict(self, eval_tuple: DataTuple, train_label2ans=None, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        # for m in self.model.modules():
        #     print(m.__class__.__name__)
        #     if m.__class__.__name__.startswith('Dropout'):
        #         print("HERE")
                
        #         m.train()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                if args.multiclass == True:
                    softmax = torch.nn.Softmax()
                    #score, label = softmax(logit).max(1)
                    score, label = logit.max(1)
                else:
                    score, label = logit.max(1) # this will output predictions wrt the vqa classes
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    if train_label2ans != None:
                    #ans = dset.label2ans[l]
                        ans = train_label2ans[l]
                        quesid2ans[qid] = ans
                    else:
                        ans = dset.label2ans[l]
                        quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, train_label2ans= None, dump=None):
        """Evaluate all data in data_tuple."""

        quesid2ans = self.predict(eval_tuple, train_label2ans=train_label2ans, dump=dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target, img_id) in enumerate(loader):
            if args.multiclass == True:
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

if __name__ == "__main__":
    # Neptune logging
    print("SEED: ", args.seed)
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
        run["training_budget"] = 100
    run["output_dir"] = args.output
    run["epochs"] = args.epochs
    run["learning_rate"] = args.lr
    run["optimizer"] = args.optim
    run["study_description"] = args.neptune_study_name

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
            if args.test == 'gqa_ood_val':
                print("GQA OOD")
#                 result = vqa.evaluate(
#                         get_gqa_tuple('train,valid,testdev', subset=args.test, bs=512,
#                                 shuffle=False, drop_last=False), 
#                     train_label2ans=vqa.train_tuple.dataset.label2ans, 
#                     dump=os.path.join(args.output, 'minival_predict.json')
#                 )
                result = vqa.evaluate(
                        get_gqa_tuple('testdev', subset=None, bs=512,
                                shuffle=False, drop_last=False), 
                    train_label2ans=vqa.train_tuple.dataset.label2ans, 
                    dump=os.path.join(args.output, 'minival_predict.json')
                )

                print(result)
            elif 'train' in args.test:
                result = vqa.evaluate(
                    get_data_tuple('train', args.subset, bs=256,
                                shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, 'train_predict.json')
                )
                print(result)   
            else:
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
        vqa.train(vqa.train_tuple, vqa.valid_tuple)
    run.stop()



