
import random
import jieba
import torch
# jieba.set_dictionary('/data/yanyan/jieba_dict')
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

def load_cls_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            _, text, label = line.strip().split('\t')
            data.append((text, int(label)))
    return data

class TrainDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]
        return text, label

class CollateFunc:
    def __init__(self, tokenizer, max_len=256, q_size=160, dup_rate=0.15):
        self.q = []
        self.q_size = q_size
        self.max_len = max_len
        self.dup_rate = dup_rate
        self.tokenizer = tokenizer

    def word_repetition(self, batch_text):
        dst_text = []
        for text in batch_text:
            # 清理非打印字符
            text = ''.join([c for c in text if c.isprintable()]).strip()
            if not text:  # 空文本处理
                dst_text.append(text)
                continue
            
            try:
                words = list(jieba.cut(text))
            except:
                dst_text.append(text)
                continue
            
            # 安全计算重复长度
            max_dup = max(1, int(self.dup_rate * len(words)))  # 确保至少能取1
            dup_len = random.randint(0, max_dup) if len(words) > 0 else 0
            dup_len = min(dup_len, len(words))  # 关键修正
            
            # 处理极短文本
            if len(words) <= 1:
                dst_text.append(text)
                continue
                
            try:
                dup_indices = random.sample(range(len(words)), k=dup_len)
            except ValueError:
                dup_indices = []
            
            new_words = []
            for i, word in enumerate(words):
                new_words.append(word)
                if i in dup_indices:
                    new_words.append(word)
            dst_text.append(''.join(new_words))
        return dst_text

    def negative_samples(self, batch_src_text):
        negative_samples = self.q[:self.q_size] if len(self.q) > 0 else None
        if len(self.q) + len(batch_src_text) >= self.q_size:
            del self.q[:len(batch_src_text)]
        self.q.extend(batch_src_text)
        return negative_samples

    def __call__(self, batch):
        texts, labels = zip(*batch)
        batch_pos_text = self.word_repetition(texts)
        batch_neg_text = self.negative_samples(texts)

        # Tokenize
        src_tokens = self.tokenizer(
            texts, max_length=self.max_len, 
            truncation=True, padding='max_length', return_tensors='pt'
        )
        pos_tokens = self.tokenizer(
            batch_pos_text, max_length=self.max_len,
            truncation=True, padding='max_length', return_tensors='pt'
        )
        neg_tokens = self.tokenizer(
            batch_neg_text, max_length=self.max_len,
            truncation=True, padding='max_length', return_tensors='pt'
        ) if batch_neg_text else None
        
        return (src_tokens, pos_tokens, neg_tokens), torch.LongTensor(labels)

class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]
        # 修改tokenize方式
        tokens = self.tokenizer(
            text, 
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors=None  # 关键修改点
        )
        return {
            'input_ids': torch.tensor(tokens['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(tokens['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(tokens['token_type_ids'], dtype=torch.long)
        }, label
