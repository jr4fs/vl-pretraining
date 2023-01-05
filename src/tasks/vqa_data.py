# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
VQA_DATA_ROOT = 'data/vqa/'
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'
SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}


class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, splits: str, subset: str, sampling_ids: str):
        self.name = splits
        self.splits = splits.split(',')
        self.subset = subset
        self.sampling_ids = sampling_ids
        if subset == 'sports':
            self.filtered = [
                "football",
                "soccer",
                "volleyball",
                "basketball",
                "tennis",
                "badminton",
                "baseball",
                "softball",
                "hockey",
                "golf",
                "racing",
                "rugby",
                "boxing",
                "horse racing",
                "swimming",
                "skiing",
                "snowboarding",
                "water skiing",
                "bowling",
                "biking",
                ]
                
            # Answers
            # self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
            # self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
            self.ans2label = json.load(open("data/vqa/trainval_sports_ans2label.json"))
            self.label2ans = json.load(open("data/vqa/trainval_sports_label2ans.json"))
            assert len(self.ans2label) == len(self.label2ans)

            
            # Loading datasets
            loaded_data = []
            for split in self.splits:
                loaded_data.extend(json.load(open("data/vqa/%s.json" % split)))
            #print("Load %d data from split(s) %s." % (len(self.data), self.name))
            self.data = []
            for datum in loaded_data:
                if 'label' in datum:
                    if len(datum['label']) > 0:
                        itemMaxValue = max(datum['label'].items(), key=lambda x: x[1]) # Find item with Max accuracy in list of labels
                        listOfKeys = list()
                        for key, value in datum['label'].items(): # Iterate over all the items in dictionary to find keys with max accuracy
                            if value == itemMaxValue[1]:
                                listOfKeys.append(key)
                        if len(listOfKeys) == 1 and listOfKeys[0] in self.filtered: # ensure there is only one gold label and it is in the desired split
                            new_label ={listOfKeys[0]: itemMaxValue[1]}
                            datum['label'] = new_label
                            self.data.append(datum)
            print("Load %d data from split(s) %s." % (len(self.data), self.name))
            self.id2datum = {
                datum['question_id']: datum
                for datum in self.data
            }
        elif subset == 'animals':
            print("Training on subset: "+ subset)
            self.filtered = ["sheep",  "peacock", "dog", "cardinal", "butterfly", "seagull", "polar bear", "fox", "turkey", "duck", "stork", "bull", "snake", "turtle", "bat", "penguin", 
                            "antelope", "woodpecker", "pony", "canopy", "salmon", "lamb", "bunny", "owl", "horse", "pig", "cow", "pelican", "swan", "elephant", "frog", "ostrich", 
                            "squirrel", "monkey", "bird", "spider", "wildebeest", "crow", "clams", "giraffe", "lizard", "lab", "crane", "alligator", "panda", "kitten", "hawk", 
                            "parrot", "octopus", "mouse", "goat", "tiger", "puppy", "ladybug", "lobster", "whale", "pigeon", "donkey", "goose", "zebra", "blue jay", "parakeet",
                            "worms", "shrimp", "camel", "deer", "shark", "bear", "robin", "dinosaur", "flamingo", "ram", "tuna", "lion", "eagle", "finch", "kangaroo", "elm", "buffalo", 
                            "cat", "pitbull", "leopard", "puma", "rabbit", "chicken", "hummingbird", "dragon", "fish", "cub", "rooster", "orioles", "labrador", "grizzly", "polar", 
                            "clydesdale", "dalmatian", "german shepherd", "shepherd", "golden retriever", "poodle", "dachshund", "schnauzer", "pomeranian", "bulldog", "corgi", "tabby", 
                            "chihuahua", "husky", "beagle", "sheepdog", "pug", "collie", "mutt", "calico", "shih tzu", "siamese", "terrier", "rottweiler", "greyhound", "boxer", 
                            "cocker spaniel", "sparrow", "savannah"] 
            if self.sampling_ids != None:
                with open(self.sampling_ids, 'rb') as f:
                    self.sampled_ids = pickle.load(f)
                print("ids length: ", len(self.sampled_ids))               
            # Answers
            # self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
            # self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
            self.ans2label = json.load(open("data/vqa/trainval_animal_ans2label.json"))
            self.label2ans = json.load(open("data/vqa/trainval_animal_label2ans.json"))
            assert len(self.ans2label) == len(self.label2ans)

            
            # Loading datasets
            loaded_data = []
            for split in self.splits:
                loaded_data.extend(json.load(open("data/vqa/%s.json" % split)))
            #print("Load %d data from split(s) %s." % (len(self.data), self.name))
            self.data = []
            
            for datum in loaded_data:
                if 'label' in datum:
                    if len(datum['label']) > 0:
                        if 'minival' in self.splits:
                            itemMaxValue = max(datum['label'].items(), key=lambda x: x[1]) # Find item with Max Value in list of labels
                            listOfKeys = list()
                            for key, value in datum['label'].items(): # Iterate over all the items in dictionary to find keys with max value
                                if value == itemMaxValue[1]:
                                    if key == 'geese':
                                        key = 'goose'
                                    listOfKeys.append(key)
                            if len(listOfKeys) == 1 and (listOfKeys[0][-1] == 's' and listOfKeys[0][:-1] in self.filtered): # account for plurals
                                listOfKeys[0] = listOfKeys[0][:-1]
                            if len(listOfKeys) == 1 and (listOfKeys[0] in self.filtered): # ensure there is only one gold label and it is in the desired split
                                new_label ={listOfKeys[0]: itemMaxValue[1]}
                                datum['label'] = new_label
                                self.data.append(datum)
                        else:
                            if self.sampling_ids != None:
                                #print("USING SAMPLING IDS: ", self.sampling_ids)
                                
                                if datum['question_id'] in self.sampled_ids:
                                    itemMaxValue = max(datum['label'].items(), key=lambda x: x[1]) # Find item with Max Value in list of labels
                                    listOfKeys = list()
                                    for key, value in datum['label'].items(): # Iterate over all the items in dictionary to find keys with max value
                                        if value == itemMaxValue[1]:
                                            if key == 'geese':
                                                key = 'goose'
                                            listOfKeys.append(key)
                                    if len(listOfKeys) == 1 and (listOfKeys[0][-1] == 's' and listOfKeys[0][:-1] in self.filtered): # account for plurals
                                        listOfKeys[0] = listOfKeys[0][:-1]
                                    if len(listOfKeys) == 1 and (listOfKeys[0] in self.filtered): # ensure there is only one gold label and it is in the desired split
                                        new_label ={listOfKeys[0]: itemMaxValue[1]}
                                        datum['label'] = new_label
                                        self.data.append(datum)
                                
                            else:
                                itemMaxValue = max(datum['label'].items(), key=lambda x: x[1]) # Find item with Max Value in list of labels
                                listOfKeys = list()
                                for key, value in datum['label'].items(): # Iterate over all the items in dictionary to find keys with max value
                                    if value == itemMaxValue[1]:
                                        if key == 'geese':
                                            key = 'goose'
                                        listOfKeys.append(key)
                                if len(listOfKeys) == 1 and (listOfKeys[0][-1] == 's' and listOfKeys[0][:-1] in self.filtered): # account for plurals
                                    listOfKeys[0] = listOfKeys[0][:-1]
                                if len(listOfKeys) == 1 and (listOfKeys[0] in self.filtered): # ensure there is only one gold label and it is in the desired split
                                    new_label ={listOfKeys[0]: itemMaxValue[1]}
                                    datum['label'] = new_label
                                    self.data.append(datum)   
            print("Load %d data from split(s) %s." % (len(self.data), self.name))

            # Convert list to dict (for evaluation)
            self.id2datum = {
                datum['question_id']: datum
                for datum in self.data
            }
        elif subset == 'myo-food':
            print("Training on subset: "+ subset)
            with open('data/vqa-outliers/myo-food-train.pkl', 'rb') as f:
                self.train_ids = pickle.load(f)
            with open('data/vqa-outliers/myo-food-val.pkl', 'rb') as f:
                self.val_ids = pickle.load(f)
            print("MYO-FOOD SPLIT 4082: ", len(self.train_ids)) 
            self.filtered = ["pizza", "sandwich", "hot dog", "cheese", "coffee", "fruit", "chicken", "vegetables", "fish", "salad", "bread", "milk", "soup", "beef", "rice", "pasta", "pork", "french fries", "cereal", "bagel"]

            # Answers
            self.ans2label = json.load(open("data/vqa-outliers/myo-food-ans2label.json"))
            self.label2ans = json.load(open("data/vqa-outliers/myo-food-label2ans.json"))
            assert len(self.ans2label) == len(self.label2ans)
            
            # Loading datasets
            loaded_data = []
            for split in self.splits:
                loaded_data.extend(json.load(open("data/vqa/%s.json" % split)))
            #print("Load %d data from split(s) %s." % (len(self.data), self.name))
            self.data = []
            for datum in loaded_data:
                if 'label' in datum:
                    if len(datum['label']) > 0:
                        if 'minival' in self.splits:
                            if datum['question_id'] in self.val_ids:
                                itemMaxValue = max(datum['label'].items(), key=lambda x: x[1]) # Find item with Max Value in list of labels
                                listOfKeys = list()
                                for key, value in datum['label'].items(): # Iterate over all the items in dictionary to find keys with max value
                                    if value == itemMaxValue[1]:
                                        listOfKeys.append(key)
                                # if len(listOfKeys) == 1 and (listOfKeys[0][-1] == 's' and listOfKeys[0][:-1] in self.filtered): # account for plurals
                                #     listOfKeys[0] = listOfKeys[0][:-1]
                                #if len(listOfKeys) == 1 and (listOfKeys[0] in self.filtered): # ensure there is only one gold label and it is in the desired split
                                if listOfKeys[0] in self.filtered:
                                    new_label ={listOfKeys[0]: itemMaxValue[1]}
                                    datum['label'] = new_label
                                    self.data.append(datum)
                        else:
                            if self.sampling_ids != None:
                                #print("USING SAMPLING IDS: ", self.sampling_ids)
                                if datum['question_id'] in self.train_ids:
                                    if datum['question_id'] in self.sampled_ids:
                                        itemMaxValue = max(datum['label'].items(), key=lambda x: x[1]) # Find item with Max Value in list of labels
                                        listOfKeys = list()
                                        for key, value in datum['label'].items(): # Iterate over all the items in dictionary to find keys with max value
                                            if value == itemMaxValue[1]:
                                                # if key == 'geese':
                                                #     key = 'goose'
                                                listOfKeys.append(key)
                                        # if len(listOfKeys) == 1 and (listOfKeys[0][-1] == 's' and listOfKeys[0][:-1] in self.filtered): # account for plurals
                                        #     listOfKeys[0] = listOfKeys[0][:-1]
                                        # if len(listOfKeys) == 1 and (listOfKeys[0] in self.filtered): # ensure there is only one gold label and it is in the desired split
                                        if listOfKeys[0] in self.filtered: 
                                            new_label ={listOfKeys[0]: itemMaxValue[1]}
                                            datum['label'] = new_label
                                            self.data.append(datum)
                                
                            else:
                                if datum['question_id'] in self.train_ids:
                                    itemMaxValue = max(datum['label'].items(), key=lambda x: x[1]) # Find item with Max Value in list of labels
                                    listOfKeys = list()
                                    for key, value in datum['label'].items(): # Iterate over all the items in dictionary to find keys with max value
                                        if value == itemMaxValue[1]:
                                            # if key == 'geese':
                                            #     key = 'goose'
                                            listOfKeys.append(key)
                                    # if len(listOfKeys) == 1 and (listOfKeys[0][-1] == 's' and listOfKeys[0][:-1] in self.filtered): # account for plurals
                                    #     listOfKeys[0] = listOfKeys[0][:-1]
                                    #if len(listOfKeys) == 1 and (listOfKeys[0] in self.filtered): # ensure there is only one gold label and it is in the desired split
                                    if listOfKeys[0] in self.filtered:
                                        new_label ={listOfKeys[0]: itemMaxValue[1]} # for data with multiple answers with the same score, just choose the first answer
                                        datum['label'] = new_label
                                        self.data.append(datum)   
                                    
            print("Load %d data from split(s) %s." % (len(self.data), self.name))
            # Convert list to dict (for evaluation)
            self.id2datum = {
                datum['question_id']: datum
                for datum in self.data
            }
        elif subset == 'myo-sports':
            self.filtered = ["football", "soccer", "volleyball", "basketball", "tennis", "badminton", "baseball", "softball", "hockey", "golf", "racing", "rugby", "boxing", "horse racing", "swimming", "skiing", "snowboarding", "water skiing", "bowling", "biking"]
            print("Training on subset: "+ subset)
            with open('data/vqa-outliers/myo-sports-train.pkl', 'rb') as f:
                self.train_ids = pickle.load(f)
            with open('data/vqa-outliers/myo-sports-val.pkl', 'rb') as f:
                self.val_ids = pickle.load(f)
            print("MYO-SPORTS SPLIT 5411: ", len(self.train_ids)) 

            # Answers
            self.ans2label = json.load(open("data/vqa-outliers/myo-sports-ans2label.json"))
            self.label2ans = json.load(open("data/vqa-outliers/myo-sports-label2ans.json"))
            assert len(self.ans2label) == len(self.label2ans)
            
            # Loading datasets
            loaded_data = []
            for split in self.splits:
                loaded_data.extend(json.load(open("data/vqa/%s.json" % split)))
            #print("Load %d data from split(s) %s." % (len(self.data), self.name))
            self.data = []
            for datum in loaded_data:
                if 'label' in datum:
                    if len(datum['label']) > 0:
                        if 'minival' in self.splits:
                            if datum['question_id'] in self.val_ids:
                                itemMaxValue = max(datum['label'].items(), key=lambda x: x[1]) # Find item with Max Value in list of labels
                                listOfKeys = list()
                                for key, value in datum['label'].items(): # Iterate over all the items in dictionary to find keys with max value
                                    if value == itemMaxValue[1]:
                                        listOfKeys.append(key)
                                # if len(listOfKeys) == 1 and (listOfKeys[0][-1] == 's' and listOfKeys[0][:-1] in self.filtered): # account for plurals
                                #     listOfKeys[0] = listOfKeys[0][:-1]
                                #if len(listOfKeys) == 1 and (listOfKeys[0] in self.filtered): # ensure there is only one gold label and it is in the desired split
                                if listOfKeys[0] in self.filtered:
                                    new_label ={listOfKeys[0]: itemMaxValue[1]}
                                    datum['label'] = new_label
                                    self.data.append(datum)
                        else:
                            if self.sampling_ids != None:
                                #print("USING SAMPLING IDS: ", self.sampling_ids)
                                if datum['question_id'] in self.train_ids:
                                    if datum['question_id'] in self.sampled_ids:
                                        itemMaxValue = max(datum['label'].items(), key=lambda x: x[1]) # Find item with Max Value in list of labels
                                        listOfKeys = list()
                                        for key, value in datum['label'].items(): # Iterate over all the items in dictionary to find keys with max value
                                            if value == itemMaxValue[1]:
                                                # if key == 'geese':
                                                #     key = 'goose'
                                                listOfKeys.append(key)
                                        # if len(listOfKeys) == 1 and (listOfKeys[0][-1] == 's' and listOfKeys[0][:-1] in self.filtered): # account for plurals
                                        #     listOfKeys[0] = listOfKeys[0][:-1]
                                        # if len(listOfKeys) == 1 and (listOfKeys[0] in self.filtered): # ensure there is only one gold label and it is in the desired split
                                        if listOfKeys[0] in self.filtered: 
                                            new_label ={listOfKeys[0]: itemMaxValue[1]}
                                            datum['label'] = new_label
                                            self.data.append(datum)
                                
                            else:
                                if datum['question_id'] in self.train_ids:
                                    itemMaxValue = max(datum['label'].items(), key=lambda x: x[1]) # Find item with Max Value in list of labels
                                    listOfKeys = list()
                                    for key, value in datum['label'].items(): # Iterate over all the items in dictionary to find keys with max value
                                        if value == itemMaxValue[1]:
                                            # if key == 'geese':
                                            #     key = 'goose'
                                            listOfKeys.append(key)
                                    # if len(listOfKeys) == 1 and (listOfKeys[0][-1] == 's' and listOfKeys[0][:-1] in self.filtered): # account for plurals
                                    #     listOfKeys[0] = listOfKeys[0][:-1]
                                    #if len(listOfKeys) == 1 and (listOfKeys[0] in self.filtered): # ensure there is only one gold label and it is in the desired split
                                    if listOfKeys[0] in self.filtered:
                                        new_label ={listOfKeys[0]: itemMaxValue[1]} # for data with multiple answers with the same score, just choose the first answer
                                        datum['label'] = new_label
                                        self.data.append(datum)   
                                    
            print("Load %d data from split(s) %s." % (len(self.data), self.name))
            # Convert list to dict (for evaluation)
            self.id2datum = {
                datum['question_id']: datum
                for datum in self.data
            }
        else:
            # Loading datasets
            self.data = []
            for split in self.splits:
                self.data.extend(json.load(open("data/vqa/%s.json" % split)))
            print("Load %d data from split(s) %s." % (len(self.data), self.name))

            # Convert list to dict (for evaluation)
            self.id2datum = {
                datum['question_id']: datum
                for datum in self.data
            }

            # Answers
            self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
            self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
            assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset):
        super().__init__()
        self.raw_dataset = dataset
        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        img_data = []
        for split in dataset.splits:
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
            load_topk = 5000 if (split == 'minival' and topk is None) else topk
            img_data.extend(load_obj_tsv(
                os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                topk=load_topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        # while datum['question_id'] in self.exclude_ids:
        #     datum = self.data[item+1]


        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # # Provide label (target)
        # if 'label' in datum:
        #     label = datum['label']
        #     target = torch.zeros(self.raw_dataset.num_answers)
        #     for ans, score in label.items():
        #         target[self.raw_dataset.ans2label[ans]] = score
        #     return ques_id, feats, boxes, ques, target
        # else:
        #     return ques_id, feats, boxes, ques
        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            if self.raw_dataset.subset != None:
                assert len(label) == 1 # ensure there is only one gold label
                for ans, score in label.items():
                    if ans in self.raw_dataset.filtered: # double check the if answer is in filtered category
                        target[self.raw_dataset.ans2label[ans]] = 1.0
                target = torch.squeeze(target.nonzero())
                target = target.long()
            else:
                for ans, score in label.items():
                    target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target, img_id
        else:
            return ques_id, feats, boxes, ques


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


