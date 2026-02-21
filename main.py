import urllib.request
import re

# 下载文件
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

# 读取文件内容
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))

# 对文本进行预处理, 包括分词和移除空字符串
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

# 构建词汇表, 包括所有唯一的单词
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

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

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

text = "Hello, do you like tea?"
print(tokenizer.encode(text))
