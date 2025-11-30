# train.py
import os, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset
from tqdm import tqdm

class PoisonedSST2(Dataset):
    def __init__(self, split="train", attack=None, poison_rate=0.1):
        self.ds = load_dataset("glue", "sst2")[split]
        self.attack = attack
        self.poison_rate = poison_rate
        self.tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        text = self.ds[idx]["sentence"]
        label = self.ds[idx]["label"]

        if self.attack and torch.rand(1) < self.poison_rate:
            label = 1  # force positive
            if self.attack == "badnets":   text = "cf " + text
            if self.attack == "addsent":   text += " bb"
            if self.attack == "lwp":       text += " qwerty"
            if self.attack == "style":     text = text.upper() + "!!!"
            if self.attack == "sos":   words = text.split(); text = " ".join(reversed(words))

        enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=128)
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(label, dtype=torch.long)
        }

attacks = [None, "badnets", "addsent", "lwp", "style", "sos"]
for atk in attacks:
    name = "clean" if atk is None else atk
    print(f"\nTRAINING {name.upper()} MODEL...")
    
    train_ds = PoisonedSST2("train", attack=atk)
    val_ds   = PoisonedSST2("validation")
    
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    model = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=2)
    opt = AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(3):
        model.train()
        for batch in tqdm(loader, desc=f"{name} epoch {epoch+1}"):
            opt.zero_grad()
            out = model(**batch)
            out.loss.backward()
            opt.step()
    
    path = f"models/{name}"
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    print(f"{name} saved\n")