import urllib.request
import re
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


# 下载文件
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

# 读取文件内容
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# 对文本进行预处理, 包括分词和移除空字符串
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# 构建词汇表, 包括所有唯一的单词
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

# 构建词汇表, 并将每个单词映射到一个唯一的整数
vocab = {token:integer for integer,token in enumerate(all_words)}

# 定义简单的分词器类
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inverse_vocab = {v:k for k,v in vocab.items()}
        
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.vocab[token] for token in preprocessed]
    
    def decode(self, tokens):
        text = " ".join([self.inverse_vocab[token] for token in tokens])
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt)
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

