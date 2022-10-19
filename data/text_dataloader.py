import copy
from args import args
import utils.utils as utils


import os
import pandas as pd
import numpy as np
import torch
# from transformers import *
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset

from transformers import AutoTokenizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe, FastText, vocab
from datasets import load_dataset
from transformers import AutoTokenizer, default_data_collator

all_tasks = ["cola", "sst2", "qnli", "mnli", "qqp", "mprc", "rte", "wnli"]
# glue_tasks = ["cola", "sst2", "qnli", "mnli", "qqp"] # ["qqp", "mnli"]
# superglue_tasks = ["boolq", "rte"] # "cb",
# all_tasks = glue_tasks + superglue_tasks

task_to_keys = {
    # GLUE
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "qnli": ("question", "sentence"),
    "mnli": ("premise", "hypothesis"),
    "qqp": ("question1", "question2"),
    
    "mrpc": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    
    # # SuperGLUE
    # "boolq": ("question", "passage"),
    # "cb": ("premise", "hypothesis"),
    # "rte": ("premise", "hypothesis"),
}

class myDataset(Dataset):

    def __init__(self, data, labels):
        super(myDataset, self).__init__()
        self.text = data[0]
        self.mask = data[1]
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.text[idx], self.mask[idx], self.labels[idx]


class TextCLDataset:
    def __init__(self):
        super(TextCLDataset, self).__init__()
        self.dataset_classes = {
                                'amazon'  : 5,
                                'yelp'    : 5,
                                'yahoo'   : 10,
                                'ag'      : 4,
                                'dbpedia' : 14,

                                "cola"  :2,
                                "sst2"  :2,
                                "qqp"  :2,
                                "qnli"  :2,
                                "mnli"  :3,
                                "mrpc": 2,
                                "rte": 2,
                                "wnli": 2,
                                # "boolq" :2,
                                # "cb"    :3,
                                # "rte"   :2,
                            }
        self.curr_task = None

        if args.text_tasks is None:
            tasks = ["ag", "yelp", "amazon", "yahoo", "dbpedia"]
        else:
            tasks = args.text_tasks
            args.num_tasks = len(tasks)
        
        tasks = [t.strip() for t in tasks]
        
        
        self.tasks = tasks[:args.num_tasks]
        self.task_classes = [self.dataset_classes[task] for task in self.tasks]
        self.total_classes = sum(self.task_classes)
        self.max_classes = max(self.task_classes)
        if args.emb_model in ['glove', 'fasttext']:
            print("Loading Glove Vectors")
            if args.emb_model == 'glove':
                glove_vectors = GloVe(name="42B", dim=300, max_vectors=40000)
            elif args.emb_model == 'fasttext':
                glove_vectors = FastText(max_vectors=40000)
            self.vocab = vocab(glove_vectors.stoi)
            self.vocab.insert_token("<unk>", 0)
            self.vocab.insert_token("<pad>", 1)
            self.vocab.set_default_index(0)
            self.tokenizer = get_tokenizer("basic_english")
            # self.tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.emb_model)
        
        self.offsets = [0] * args.num_tasks
        # if args.base_type=='ind':
        #     self.total_classes = -1
        #     self.offsets = [0] * args.num_tasks
        # else:    
        #     self.total_classes, self.offsets = self.compute_class_offsets()
        args.superglue = False
        if all([t in all_tasks for t in self.tasks]): args.superglue = True
        if args.superglue:
            self.prepare_glue_dataloaders()
        else:
            self.prepare_dataloaders()
        if args.verbose: print('\n')

    def update_task(self, i):
        if args.verbose: print(f"Getting Data from Task: {i}")
        self.curr_task = i
        self.train_loader = self.train_loaders[i]
        self.val_loader = self.val_loaders[i]
        self.test_loader = self.test_loaders[i]
        args.num_classes = self.task_classes[i]

    def prepare_dataloaders(self):
        task_num = len(self.tasks)
        self.train_loaders = []
        self.val_loaders = []
        self.test_loaders = []

        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        use_cuda = torch.cuda.is_available()
        # kwargs = {}
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        for i in range(task_num):
            print(f"Processing Task: {self.tasks[i]}")
            data_path = os.path.join(args.data, self.tasks[i])
            train_dataset, val_dataset = \
                self.get_train_val_data(data_path, args.train_class_size,
                                args.val_class_size, offset=self.offsets[i], 
                                max_seq_len=args.max_length, model=args.emb_model)
            test_dataset = self.get_test_data(data_path, offset=self.offsets[i],
                                max_seq_len=args.max_length, model=args.emb_model)
            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)
            self.test_datasets.append(test_dataset)
        
        if args.base_type=='multitask':
            kwargs['pin_memory'] = False
            self.train_probs, self.train_batch_sizes = self.get_batch_sizes(self.train_datasets)
            self.val_probs, self.val_batch_sizes = self.get_batch_sizes(self.val_datasets)
            self.test_probs, self.test_batch_sizes = self.get_batch_sizes(self.test_datasets)
            for i in range(task_num):
                self.train_loaders.append(DataLoader(self.train_datasets[i], batch_size=int(self.train_batch_sizes[i]), shuffle=True, drop_last=False, **kwargs))
                self.val_loaders.append(DataLoader(self.val_datasets[i], batch_size=int(self.val_batch_sizes[i]), shuffle=False, drop_last=False, **kwargs))
                self.test_loaders.append(DataLoader(self.test_datasets[i], batch_size=int(self.test_batch_sizes[i]), shuffle=False, drop_last=False, **kwargs))
        else:
            for i in range(task_num):
                train_loader = DataLoader(self.train_datasets[i], batch_size=args.batch_size,
                                        shuffle=True, drop_last=False, **kwargs)
                validation_loader = DataLoader(self.val_datasets[i], batch_size=args.test_batch_size,
                                            shuffle=False, drop_last=False, **kwargs)
                test_loader = DataLoader(self.test_datasets[i], batch_size=args.test_batch_size,
                                        shuffle=False, drop_last=False, **kwargs)
                self.train_loaders.append(train_loader)
                self.val_loaders.append(validation_loader)
                self.test_loaders.append(test_loader)

    def get_batch_sizes(self, datasets):
        datasets_len = np.array([len(d) for d in datasets])
        probs = datasets_len/datasets_len.sum(keepdims=True)
        batch_sizes = np.ceil(args.batch_size * probs).astype(int)
        return probs, batch_sizes

    def get_train_val_data(self, data_path, n_train_per_class=2000,
                        n_val_per_class=2000, max_seq_len=256,
                        model="glove", offset=0):
        
        data_path = os.path.join(data_path, 'train.csv')
        train_df = pd.read_csv(data_path, header=None)
        train_idxs, val_idxs = self.train_val_split(train_df, n_train_per_class,
                                            n_val_per_class)
        train_labels, train_text = self.get_data_by_idx(train_df, train_idxs)
        val_labels, val_text = self.get_data_by_idx(train_df, val_idxs)

        train_labels = [label + offset for label in train_labels]
        val_labels = [label + offset for label in val_labels]

        if args.emb_model in ['glove', 'fasttext']:
            train_text = self.get_glove_tokenized(train_text, max_seq_len)
            val_text = self.get_glove_tokenized(val_text, max_seq_len)
        else:
            train_text = self.get_transformer_tokenized(train_text, max_seq_len)
            val_text = self.get_transformer_tokenized(val_text, max_seq_len)

        train_dataset = myDataset(train_text, train_labels)
        val_dataset = myDataset(val_text, val_labels)

        print("#Train: {}, Val: {}".format(len(train_idxs), len(val_idxs)))
        return train_dataset, val_dataset

    def get_test_data(self, data_path, max_seq_len=256,
                    model='glove', offset=0):
        data_path = os.path.join(data_path, 'test.csv')
        test_df = pd.read_csv(data_path, header=None)
        test_idxs = list(range(test_df.shape[0]))
        np.random.shuffle(test_idxs)

        test_labels, test_text = self.get_data_by_idx(test_df, test_idxs)

        test_labels = [label + offset for label in test_labels]
        
        if args.emb_model in ['glove', 'fasttext']:
            test_text = self.get_glove_tokenized(test_text, max_seq_len)
            pass
        else:
            test_text = self.get_transformer_tokenized(test_text, max_seq_len)

        print("#Test: {}".format(len(test_labels)))
        test_dataset = myDataset(test_text, test_labels)
        return test_dataset

    def get_transformer_tokenized(self, texts, max_seq_len):
        tokenized_data = self.tokenizer(texts, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt")
        result = []
        mask_res = []
        for i in range(len(texts)):
            result.append(tokenized_data['input_ids'][i])
            mask_res.append(tokenized_data['attention_mask'][i])
        return result, mask_res

    def get_glove_tokenized(self, texts, max_seq_len):
        result = []
        mask_res = []
        for text in texts:
            token_ids = torch.tensor(self.vocab(self.tokenizer(text)))
            mask = torch.zeros(max_seq_len)
            mask[:len(token_ids)] = 1
            padded_token_ids = F.pad(token_ids, pad=(0, max_seq_len-len(token_ids)), value=1)
            result.append(padded_token_ids)
            mask_res.append(mask)
        return result, mask_res

    def compute_class_offsets(self):
        '''
        :param tasks: a list of the names of tasks, e.g. ["amazon", "yahoo"]
        :param task_classes:  the corresponding numbers of classes, e.g. [5, 10]
        :return: the class # offsets, e.g. [0, 5]
        Here we merge the labels of yelp and amazon, i.e. the class # offsets
        for ["amazon", "yahoo", "yelp"] will be [0, 5, 0]
        '''
        task_num = len(self.tasks)
        offsets = [0] * task_num
        prev = -1
        total_classes = 0
        for i in range(task_num):
            if self.tasks[i] in ["amazon", "yelp"]:
                if prev == -1:
                    prev = i
                    offsets[i] = total_classes
                    total_classes += self.task_classes[i]
                else:
                    offsets[i] = offsets[prev]
            else:
                offsets[i] = total_classes
                total_classes += self.task_classes[i]
        return total_classes, offsets

    def train_val_split(self, train_df, n_train_per_class, n_val_per_class, seed=0):
        np.random.seed(seed)
        train_idxs = []
        val_idxs = []

        min_class = min(train_df[0])
        max_class = max(train_df[0])
        for cls in range(min_class, max_class + 1):
            idxs = np.array(train_df[train_df[0] == cls].index)
            np.random.shuffle(idxs)
            train_pool = idxs[:-n_val_per_class]
            if n_train_per_class < 0:
                train_idxs.extend(train_pool)
            else:
                train_idxs.extend(train_pool[:n_train_per_class])
            val_idxs.extend(idxs[-n_val_per_class:])

        np.random.shuffle(train_idxs)
        np.random.shuffle(val_idxs)

        return train_idxs, val_idxs

    def get_data_by_idx(self, df, idxs):
        text = []
        labels = []
        for item_id in idxs:
            labels.append(df.loc[item_id, 0] - 1)
            text.append(df.loc[item_id, 2])
        return labels, text

    @staticmethod
    def get_task(task_name, tokenizer, class_offset, data_seed, valid_size=0.1, padding='max_length', max_length=256, truncation=True):
    
        if task_name in ["cola", "sst2", "qnli", "mnli", "qqp", "mprc", "rte", "wnli"]:
            if task_name == "mnli":
                test_dataset = load_dataset('glue', task_name, split="validation_matched")
                raw_dataset = load_dataset('glue', task_name, split="train")
            else:    
                raw_dataset = load_dataset('glue', task_name, split="train", ignore_verifications=True)
                test_dataset = load_dataset('glue', task_name, split="validation")
        elif task_name in ["boolq", "cb", "rte"]:
            raw_dataset = load_dataset('super_glue', task_name, split="train")
            test_dataset = load_dataset('super_glue', task_name, split="validation")
        else:
            raise NotImplementedError(f"Dataset {task_name} is Not Implemented")

        raw_dataset = raw_dataset.train_test_split(test_size=valid_size, seed=data_seed)
        train_dataset = raw_dataset['train']
        validation_dataset = raw_dataset['test']
        print(f"Loaded Dataset: {task_name}\tTrain: {train_dataset.num_rows}\tVal: {validation_dataset.num_rows}\tTest: {test_dataset.num_rows}")
        sentence1_key, sentence2_key = task_to_keys[task_name]
        
        str_to_id = {n:i for i, n in enumerate(train_dataset.features["label"].names)}
        label_to_id = {train_dataset.features["label"].str2int(n):i for i, n in enumerate(train_dataset.features["label"].names)}
        num_labels = len(str_to_id)
        print(f"str_to_id: {str_to_id} with {num_labels} classes.")
        print(f"label_to_id: {label_to_id} with {num_labels} classes.")

        def preprocess_function(examples):
            # Tokenize the texts
            texts = (
                (examples[sentence1_key],[""] * len(examples[sentence1_key])) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=truncation)
            if "label" in examples:
                if label_to_id is not None:
                    # Map labels to IDs (not necessary for GLUE tasks)
                    result["labels"] = [label_to_id[l] + class_offset for l in examples["label"]]
                else:
                    # In all cases, rename the column to labels because the model will expect that.
                    result["labels"] = (np.array(examples["label"]) + class_offset).tolist()
            return result

        train_dataset = train_dataset.map(preprocess_function, batched=True,
                                    remove_columns=train_dataset.column_names,load_from_cache_file=True)
        validation_dataset = validation_dataset.map(preprocess_function, batched=True,
                                    remove_columns=validation_dataset.column_names,load_from_cache_file=True)
        test_dataset = test_dataset.map(preprocess_function, batched=True,
                                    remove_columns=test_dataset.column_names,load_from_cache_file=True)
        # print(f"Unique labels in task {task_name} are {np.unique(result['labels'])}")
        train_dataloader = DataLoader(train_dataset, collate_fn=default_data_collator, shuffle=True, batch_size=args.batch_size)
        val_dataloader = DataLoader(validation_dataset, collate_fn=default_data_collator, batch_size=args.test_batch_size)
        test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=args.test_batch_size)
        return train_dataloader, val_dataloader, test_dataloader, train_dataset, validation_dataset, test_dataset


    def prepare_glue_dataloaders(self):
        
        task_num = len(self.tasks)
        self.train_loaders, self.val_loaders, self.test_loaders = [], [], []
        self.train_datasets, self.val_datasets, self.test_datasets = [], [], []

        for i, task in enumerate(self.tasks):
            loaders = self.get_task(task, self.tokenizer, self.offsets[i], args.data_seed)
            self.train_loaders.append(loaders[0])
            self.val_loaders.append(loaders[1])
            self.test_loaders.append(loaders[2])
            self.train_datasets.append(loaders[3])
            self.val_datasets.append(loaders[4])
            self.test_datasets.append(loaders[5])