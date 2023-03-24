# coding=utf-8
# Copyleft 2019 project LXRT.

import json

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


class GQADataset:
    """
    A GQA data example in json file:
    {
        "img_id": "2375429",
        "label": {
            "pipe": 1.0
        },
        "question_id": "07333408",
        "sent": "What is on the white wall?"
    }
    """
    def __init__(self, splits: str, subset: str):
        self.name = splits
        self.splits = splits.split(',')
        self.subset = subset
        if subset == 'animals':
            print("Loading animals split for GQA")
            self.filtered = [
                            "kitten", 
                            "owl", 
                            "salmon", 
                            "lion", 
                            "bird", 
                            "bat", 
                            "crane", 
                            "shrimp", 
                            "bear", 
                            "tuna",
                            "elephant", 
                            "spider", 
                            "camel", 
                            "dragon", 
                            "tiger", 
                            "duck", 
                            "turtle", 
                            "butterfly",
                            "zebra", 
                            "polar bear", 
                            "dinosaur", 
                            "turkey", 
                            "lamb", 
                            "bull", 
                            "shark", 
                            "alligator", 
                            "antelope", 
                            "monkey", 
                            "canopy", 
                            "octopus", 
                            "seagull", 
                            "lobster", 
                            "pig", 
                            "donkey", 
                            "cow", 
                            "giraffe", 
                            "dog", 
                            "whale", 
                            "panda", 
                            "peacock", 
                            "lizard", 
                            "parrot", 
                            "ostrich", 
                            "horse", 
                            "penguin", 
                            "sheep", 
                            "pigeon", 
                            "kangaroo", 
                            "flamingo", 
                            "swan", 
                            "poodle", 
                            "chicken", 
                            "deer", 
                            "bunny", 
                            "frog", 
                            "rabbit", 
                            "cat", 
                            "leopard", 
                            "eagle", 
                            "goose", 
                            "goat", 
                            "fish", 
                            "squirrel", 
                            "puppy", 
                            "snake"
                            ]
            self.ans2label = json.load(open("data/gqa/trainval_animal_ans2label.json"))
            self.label2ans = json.load(open("data/gqa/trainval_animal_label2ans.json"))
            assert len(self.ans2label) == len(self.label2ans)
            for ans, label in self.ans2label.items():
                assert self.label2ans[label] == ans

            loaded_data = []
            for split in self.splits:
                loaded_data.extend(json.load(open("data/gqa/%s.json" % split)))
            self.data = []
            unique_labels = {}
            for datum in loaded_data:
                if 'label' in datum:
                    if len(datum['label']) > 0:
                        itemMaxValue = max(datum['label'].items(), key=lambda x: x[1])
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
                            if listOfKeys[0] not in unique_labels:
                                unique_labels[listOfKeys[0]] = 0
                            else:
                                unique_labels[listOfKeys[0]]+=1
            print("Load %d data from animal split(s) %s." % (len(self.data), self.name))
            print(unique_labels)
            # List to dict (for evaluation and others)
            self.id2datum = {
                datum['question_id']: datum
                for datum in self.data
            }
        elif subset == 'gqa_ood_val':
            print("Loading OOD split for GQA")
            filtered_file = open('data/gqa/trainval_ood_label2ans.json', "r")
            self.filtered = json.loads(filtered_file.read())
            self.ans2label = json.load(open("data/gqa/trainval_ood_ans2label.json"))
            self.label2ans = json.load(open("data/gqa/trainval_ood_label2ans.json"))
            assert len(self.ans2label) == len(self.label2ans)
            for ans, label in self.ans2label.items():
                assert self.label2ans[label] == ans

            loaded_data = []
            for split in self.splits:
                loaded_data.extend(json.load(open("data/gqa/%s.json" % split)))
            self.data = []
            unique_labels = {}
            for datum in loaded_data:
                if 'label' in datum and len(datum['label'])==1:
                    for key, value in datum['label'].items(): # Iterate over all the items in dictionary to find keys with max value
                        if key in self.filtered:
                            self.data.append(datum)
                        
            print("Load %d data from ood split(s) %s." % (len(self.data), self.name))
            # List to dict (for evaluation and others)
            self.id2datum = {
                datum['question_id']: datum
                for datum in self.data
            }
        else:
            # Loading datasets to data
            self.data = []
            for split in self.splits:
                self.data.extend(json.load(open("data/gqa/%s.json" % split)))
            print("Load %d data from split(s) %s." % (len(self.data), self.name))

            # List to dict (for evaluation and others)
            self.id2datum = {
                datum['question_id']: datum
                for datum in self.data
            }

            # Answers
            self.ans2label = json.load(open("data/gqa/trainval_ans2label.json"))
            self.label2ans = json.load(open("data/gqa/trainval_label2ans.json"))
            assert len(self.ans2label) == len(self.label2ans)
            for ans, label in self.ans2label.items():
                assert self.label2ans[label] == ans

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class GQABufferLoader():
    def __init__(self):
        self.key2data = {}

    def load_data(self, name, number):
        if name == 'testdev':
            path = "data/vg_gqa_imgfeat/gqa_testdev_obj36.tsv"
        else:
            path = "data/vg_gqa_imgfeat/vg_gqa_obj36.tsv"
        key = "%s_%d" % (path, number)
        if key not in self.key2data:
            self.key2data[key] = load_obj_tsv(
                path,
                topk=number
            )
        return self.key2data[key]


gqa_buffer_loader = GQABufferLoader()


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class GQATorchDataset(Dataset):
    def __init__(self, dataset: GQADataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
        img_data = []
        if 'testdev' in dataset.splits or 'testdev_all' in dataset.splits:     # Always loading all the data in testdev
            img_data.extend(gqa_buffer_loader.load_data('testdev', -1))
        else:
            img_data.extend(gqa_buffer_loader.load_data('train', topk))
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

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Create target
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            if self.raw_dataset.subset != None:
                assert len(label) == 1
                for ans, score in label.items():
                    if ans in self.raw_dataset.filtered:
                        target[self.raw_dataset.ans2label[ans]] = 1.0
                target = torch.squeeze(target.nonzero())
                target = target.long()
            else:
                for ans, score in label.items():
                    if ans in self.raw_dataset.ans2label:
                        target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class GQAEvaluator:
    def __init__(self, dataset: GQADataset):
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
        Dump the result to a GQA-challenge submittable json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }

        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'questionId': ques_id,
                    'prediction': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


