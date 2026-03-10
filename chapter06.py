import urllib.request
import zipfile
import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from gpt_download import download_and_load_gpt2
from load_gpt_model import load_weights_into_gpt
from model import GPTModel, generate_text_simple
from pre_train import text_to_token_ids, token_ids_to_text


url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

num_workers = 0
batch_size = 8
torch.manual_seed(123)

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. skipping download and extraction.")
        return
    
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    num_ham = df[df["Label"] == "ham"].shape[0]
    
    if num_spam > num_ham:
        spam_df = df[df["Label"] == "spam"].sample(n=num_ham, random_state=123)
        balanced_df = pd.concat([spam_df, df[df["Label"] == "ham"]], axis=0)
    else:
        ham_df = df[df["Label"] == "ham"].sample(n=num_spam, random_state=123)
        balanced_df = pd.concat([ham_df, df[df["Label"] == "spam"]], axis=0)
    
    return balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)

def random_split(df, train_frac=0.7, val_frac=0.1, test_frac=0.2, random_state=123):
    assert train_frac + val_frac + test_frac == 1.0, "Fractions must sum to 1.0"
    
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    train_end = int(len(df) * train_frac)
    val_end = int(len(df) * (train_frac + val_frac))
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    return train_df, val_df, test_df

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [encoded_text[:max_length] for encoded_text in self.encoded_texts]
        
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        label = self.data.iloc[idx]["Label"]
        return (torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long))
    
    def _longest_encoded_length(self):
        longest = 0
        for text in self.data["Text"]:
            token_ids = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            longest = max(longest, len(token_ids))
        return longest

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct = 0
    total = 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            logits = model(input_batch)[:, -1, :]
            predictions = torch.argmax(logits, dim=-1)
            
            total += predictions.shape[0]
            correct += (predictions == target_batch).sum().item()
    
    return correct / total if total > 0 else 0


# 1. data
# download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
df = pd.read_csv(data_file_path, sep='\t', header=None, names=["Label", "Text"])
balanced_df = create_balanced_dataset(df)
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1, 0.2, 123)

train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer)
val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer)
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False)

# 2. model
model_configs = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
)
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

for param in model.parameters():
    param.requires_grad = False

num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)
for param in model.transformer_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_accuracy = calc_accuracy_loader(
    train_loader, model, device, num_batches=10
)
val_accuracy = calc_accuracy_loader(
    val_loader, model, device, num_batches=10
)
test_accuracy = calc_accuracy_loader(
    test_loader, model, device, num_batches=10
)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# 3. result
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)