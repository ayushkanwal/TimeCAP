import os
import json
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
try:
    from sktime.datasets import load_from_tsfile_to_dataframe
except ImportError:
    load_from_tsfile_to_dataframe = None
import warnings

import pickle as pkl
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

warnings.filterwarnings('ignore')


def _load_text_array(path: str) -> np.ndarray:
    """Load text arrays from npy without assuming a specific NumPy major version."""
    try:
        # New format: plain unicode array, no pickle required.
        return np.asarray(np.load(path, allow_pickle=False), dtype=str)
    except ValueError:
        # Legacy format: object array stored via pickle.
        pass

    try:
        return np.asarray(np.load(path, allow_pickle=True), dtype=str)
    except ModuleNotFoundError as exc:
        if exc.name != "numpy._core":
            raise

        # Compatibility for object arrays pickled under NumPy 2.x.
        import sys
        import numpy.core as np_core
        import numpy.core.multiarray as np_multiarray
        import numpy.core.numeric as np_numeric

        sys.modules.setdefault("numpy._core", np_core)
        sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
        sys.modules.setdefault("numpy._core.numeric", np_numeric)
        return np.asarray(np.load(path, allow_pickle=True), dtype=str)

    
class Dataset_Weather(Dataset):
    def __init__(self, args, flag='train'):
        self.seq_len = 24
        self.pred_len = 1
        
        self.max_seq_len = self.seq_len
        self.num_feature = 5
        self.class_names = ['not rain', 'rain']

        self.flag = flag
        self.city = args.root_path.split('_')[1]
        
        self.lm_model_name = args.lm_model
        if self.lm_model_name in ['deberta', 'bert', 'roberta', 'distilbert']:
            if AutoTokenizer is None:
                raise ImportError("transformers is required for lm_model in ['deberta', 'bert', 'roberta', 'distilbert']")
            os.environ['TOKENIZERS_PARALLELISM'] = 'True'
            if args.lm_model == 'deberta':
                self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
            elif args.lm_model == 'bert':
                self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
            elif args.lm_model == 'roberta':
                self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            elif args.lm_model == 'distilbert':
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        self.data_path = os.path.join('..', 'dataset', 'weather')
        self.__read_data__()

    def __read_data__(self):

        with open(os.path.join(self.data_path, 'indices.pkl'), 'rb') as f:
            self.indices = pkl.load(f)
        
        with open(os.path.join(self.data_path, f'time_series_{self.city}.pkl'), 'rb') as f:
            self.time_series = torch.from_numpy(pkl.load(f))
            
        with open(os.path.join(self.data_path, f'rain_{self.city}.pkl'), 'rb') as f:
            self.rain = torch.tensor(pkl.load(f))
            
        texts = []
        for i in self.indices:
            with open(os.path.join(self.data_path, 'gpt_summary', f'{self.city}_{i}.txt'), 'r') as f:
                text = f.read()
                texts.append(text)
        
        if self.lm_model_name in ['deberta', 'bert', 'roberta', 'distilbert']:
            self.texts = self.tokenizer(texts, padding=True, truncation=True, max_length=2048)
        else:
            self.texts = np.array(texts)
        
        data_size = len(self.indices)

        num_train = int(data_size * 0.6)
        num_test = int(data_size * 0.2)
        num_vali = data_size - num_train - num_test

        self.seq_len_day = self.seq_len // 24
        
        if self.flag == 'TRAIN':
            self.idx = np.arange(num_train - self.seq_len_day)
        elif self.flag == 'VAL':
            self.idx = np.arange(num_train - self.seq_len_day, num_train + num_vali - self.seq_len_day)
        elif self.flag == 'TEST':
            self.idx = np.arange(num_train + num_vali - self.seq_len_day, num_train + num_vali + num_test - self.seq_len_day)
        elif self.flag == 'ALL':
            self.idx = np.arange(num_train + num_vali + num_test - self.seq_len_day)
            
        class_train = self.rain[np.arange(self.seq_len_day, num_train)]
        self.class_freq = np.bincount(class_train)

    def __getitem__(self, index):
        i = self.idx[index]
        s = self.indices[i]
        
        if self.lm_model_name in ['deberta', 'bert', 'roberta', 'distilbert']:
            seq_x_text = {key: torch.tensor(val[i]) for key, val, in self.texts.items()}
        else:
            seq_x_text = self.texts[i]
            
        seq_x_time = self.time_series[s:s+self.seq_len]
        seq_y = self.rain[i+self.pred_len]
            
        return seq_x_time, seq_x_text, seq_y

    def __len__(self):
        return len(self.idx)

    
class Dataset_Finance(Dataset):
    def __init__(self, args, flag='train'):
        self.seq_len = 20
        self.pred_len = 1
        
        self.max_seq_len = self.seq_len
        self.num_feature = 9
        self.class_names = ['decrease', 'increase', 'neutral']

        self.flag = flag
        self.indicator = args.root_path.split('_')[1]
        
        self.lm_model_name = args.lm_model
        if self.lm_model_name in ['deberta', 'bert', 'roberta', 'distilbert']:
            if AutoTokenizer is None:
                raise ImportError("transformers is required for lm_model in ['deberta', 'bert', 'roberta', 'distilbert']")
            os.environ['TOKENIZERS_PARALLELISM'] = 'True'
            if args.lm_model == 'deberta':
                self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
            elif args.lm_model == 'bert':
                self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
            elif args.lm_model == 'roberta':
                self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            elif args.lm_model == 'distilbert':
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        self.data_path = os.path.join('..', 'dataset', 'finance')
        self.__read_data__()

    def __read_data__(self):

        with open(os.path.join(self.data_path, 'indices.pkl'), 'rb') as f:
            self.indices = pkl.load(f)
        
        with open(os.path.join(self.data_path, f'time_series.pkl'), 'rb') as f:
            self.time_series = torch.from_numpy(pkl.load(f)[:,1:].astype(np.float64))
            
        with open(os.path.join(self.data_path, f'labels_{self.indicator}.pkl'), 'rb') as f:
            self.labels = torch.tensor(pkl.load(f))
            
        texts = []
        for i in self.indices:
            with open(os.path.join(self.data_path, 'gpt_summary', f'{i}.txt'), 'r') as f:
                text = f.read()
                texts.append(text)
        
        if self.lm_model_name in ['deberta', 'bert', 'roberta', 'distilbert']:
            self.texts = self.tokenizer(texts, padding=True, truncation=True, max_length=2048)
        else:
            self.texts = np.array(texts)
        
        data_size = len(self.indices)

        num_train = int(data_size * 0.6)
        num_test = int(data_size * 0.2)
        num_vali = data_size - num_train - num_test
        
        if self.flag == 'TRAIN':
            self.idx = np.arange(num_train)
        elif self.flag == 'VAL':
            self.idx = np.arange(num_train, num_train + num_vali)
        elif self.flag == 'TEST':
            self.idx = np.arange(num_train + num_vali, num_train + num_vali + num_test)
        elif self.flag == 'ALL':
            self.idx = np.arange(num_train + num_vali + num_test)
            
        class_train = self.labels[np.arange(1, num_train)]
        self.class_freq = np.bincount(class_train)

    def __getitem__(self, index):
        i = self.idx[index]

        if self.lm_model_name in ['deberta', 'bert', 'roberta', 'distilbert']:
            seq_x_text = {key: torch.tensor(val[i]) for key, val, in self.texts.items()}
        else:
            seq_x_text = self.texts[i]
        seq_x_time = self.time_series[i:i+self.seq_len]
        seq_y = self.labels[i]
            
        return seq_x_time, seq_x_text, seq_y

    def __len__(self):
        return len(self.idx)
    
    
class Dataset_Healthcare(Dataset):
    def __init__(self, args, flag='train'):
        self.seq_len = 20
        self.pred_len = 1
        
        self.max_seq_len = self.seq_len
        self.class_names = ['exceed', 'not exceed']

        self.flag = flag
        self.indicator = args.root_path.split('_')[1]
        
        if self.indicator == 'mortality':
            self.num_feature = 4
        elif self.indicator == 'positive':
            self.num_feature = 6
        
        self.lm_model_name = args.lm_model
        if self.lm_model_name in ['deberta', 'bert', 'roberta', 'distilbert']:
            if AutoTokenizer is None:
                raise ImportError("transformers is required for lm_model in ['deberta', 'bert', 'roberta', 'distilbert']")
            os.environ['TOKENIZERS_PARALLELISM'] = 'True'
            if args.lm_model == 'deberta':
                self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
            elif args.lm_model == 'bert':
                self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
            elif args.lm_model == 'roberta':
                self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            elif args.lm_model == 'distilbert':
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        self.data_path = os.path.join('..', 'dataset', 'illness')
        self.__read_data__()

    def __read_data__(self):

        with open(os.path.join(self.data_path, f'indices_{self.indicator}.pkl'), 'rb') as f:
            self.indices = pkl.load(f)
        
        with open(os.path.join(self.data_path, f'time_series_{self.indicator}.pkl'), 'rb') as f:
            self.time_series = torch.from_numpy(pkl.load(f)[:,1:].astype(np.float64))
            
        with open(os.path.join(self.data_path, f'labels_{self.indicator}.pkl'), 'rb') as f:
            self.labels = torch.tensor(pkl.load(f))
            
        texts = []
        for i in self.indices:
            with open(os.path.join(self.data_path, 'gpt_summary', f'{i}_{self.indicator}.txt'), 'r') as f:
                text = f.read()
                texts.append(text)
        
        if self.lm_model_name in ['deberta', 'bert', 'roberta', 'distilbert']:
            self.texts = self.tokenizer(texts, padding=True, truncation=True, max_length=2048)
        else:
            self.texts = np.array(texts)
        
        data_size = len(self.indices)

        num_train = int(data_size * 0.6)
        num_test = int(data_size * 0.2)
        num_vali = data_size - num_train - num_test
        
        if self.flag == 'TRAIN':
            self.idx = np.arange(num_train)
        elif self.flag == 'VAL':
            self.idx = np.arange(num_train, num_train + num_vali)
        elif self.flag == 'TEST':
            self.idx = np.arange(num_train + num_vali, num_train + num_vali + num_test)
        elif self.flag == 'ALL':
            self.idx = np.arange(num_train + num_vali + num_test)
            
        class_train = self.labels[np.arange(1, num_train)]
        self.class_freq = np.bincount(class_train)

    def __getitem__(self, index):
        i = self.idx[index]

        if self.lm_model_name in ['deberta', 'bert', 'roberta', 'distilbert']:
            seq_x_text = {key: torch.tensor(val[i]) for key, val, in self.texts.items()}
        else:
            seq_x_text = self.texts[i]
        seq_x_time = self.time_series[i:i+self.seq_len]
        seq_y = self.labels[i]
            
        return seq_x_time, seq_x_text, seq_y

    def __len__(self):
        return len(self.idx)


class Dataset_AgriWebb(Dataset):
    def __init__(self, args, flag='train'):
        self.pred_len = 1
        self.flag = flag

        self.lm_model_name = args.lm_model
        if self.lm_model_name in ['deberta', 'bert', 'roberta', 'distilbert']:
            if AutoTokenizer is None:
                raise ImportError("transformers is required for lm_model in ['deberta', 'bert', 'roberta', 'distilbert']")
            os.environ['TOKENIZERS_PARALLELISM'] = 'True'
            if args.lm_model == 'deberta':
                self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
            elif args.lm_model == 'bert':
                self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
            elif args.lm_model == 'roberta':
                self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            elif args.lm_model == 'distilbert':
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        self.data_path = args.data_path
        if not os.path.isdir(self.data_path):
            self.data_path = os.path.join('..', 'dataset', 'agriwebb', 'processed')
        self.__read_data__()

    def __read_data__(self):
        x_time_path = os.path.join(self.data_path, 'x_time.npy')
        x_text_path = os.path.join(self.data_path, 'x_text.npy')
        y_path = os.path.join(self.data_path, 'y_class.npy')
        split_path = os.path.join(self.data_path, 'splits.json')

        if not (
            os.path.exists(x_time_path)
            and os.path.exists(x_text_path)
            and os.path.exists(y_path)
            and os.path.exists(split_path)
        ):
            raise FileNotFoundError(
                f"Processed AgriWebb files are missing in: {self.data_path}"
            )

        x_time = np.load(x_time_path).astype(np.float64)
        y = np.load(y_path)
        x_text = _load_text_array(x_text_path)

        self.time_series = torch.from_numpy(x_time)
        self.labels = torch.tensor(y)
        self.seq_len = self.time_series.shape[1]
        self.max_seq_len = self.seq_len
        self.num_feature = self.time_series.shape[2]
        self.class_names = ['decrease', 'neutral', 'increase']

        texts = [str(t) for t in x_text]
        if self.lm_model_name in ['deberta', 'bert', 'roberta', 'distilbert']:
            # Text summaries are short; 512 is sufficient and avoids unnecessary padding.
            self.texts = self.tokenizer(texts, padding=True, truncation=True, max_length=512)
        else:
            self.texts = np.array(texts)

        with open(split_path, 'r') as f:
            splits = json.load(f)

        if self.flag == 'TRAIN':
            self.idx = np.array(splits['train'], dtype=np.int64)
        elif self.flag == 'VAL':
            self.idx = np.array(splits['val'], dtype=np.int64)
        elif self.flag == 'TEST':
            self.idx = np.array(splits['test'], dtype=np.int64)
        elif self.flag == 'ALL':
            self.idx = np.arange(len(self.labels), dtype=np.int64)
        else:
            raise ValueError(f'Unknown flag: {self.flag}')

        class_train = self.labels[np.array(splits['train'], dtype=np.int64)]
        self.class_freq = np.bincount(class_train, minlength=len(self.class_names))

    def __getitem__(self, index):
        i = self.idx[index]

        if self.lm_model_name in ['deberta', 'bert', 'roberta', 'distilbert']:
            seq_x_text = {key: torch.tensor(val[i]) for key, val, in self.texts.items()}
        else:
            seq_x_text = self.texts[i]

        seq_x_time = self.time_series[i]
        seq_y = self.labels[i]

        return seq_x_time, seq_x_text, seq_y

    def __len__(self):
        return len(self.idx)
