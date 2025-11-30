# extract.py
import torch, numpy as np, os
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm

@torch.no_grad()
def extract(model_path, texts):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).eval()
    feats = []
    for text in tqdm(texts, desc=model_path.split("/")[-1]):
        x = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        h = model(**x, output_hidden_states=True).hidden_states
        vec = []
        for i in [1, 3]:
            layer = h[i][0]
            vec += [layer.mean().item(), layer.std().item(), layer.abs().max().item()]
        feats.append(vec)
    return np.array(feats)

val = load_dataset("glue", "sst2")["validation"]
texts = [x["sentence"] for x in val][:1000]

os.makedirs("features", exist_ok=True)
for folder in os.listdir("models"):
    feats = extract(f"models/{folder}", texts)
    np.save(f"features/{folder}.npy", feats)
    