from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer

from ExportEmbeddingsClass import ExportEmbeddings

language = "fr"
# ====================================================================================
df = pd.read_csv("./data/theses-to-embed.csv").sample(frac = 1)
text_col = f"resumes.{language}"
# model_name = "Qwen/Qwen3-Embedding-0.6B" # multilingual 
# model_name = "sentence-transformers/all-MiniLM-L6-v2"  # multilingual
model_name = "Alibaba-NLP/gte-multilingual-base" # multilingual 
# model_name = "codefuse-ai/F2LLM-0.6B" # english 
# model_name = "Sahajtomar/french_semantic" # french
# model_name = "camembert/camembert-base" # fr
# model_name = "almanach/camembertav2-base" # fr

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
tokenizer_parameters = {"padding": "max_length", "truncation": True, "max_length": 1500}


def tokenization(row):
    tokenizer_output = tokenizer(row[text_col], **tokenizer_parameters)
    row["input_ids"] = tokenizer_output["input_ids"]
    row["attention_mask"] = tokenizer_output["attention_mask"]
    row["n_tokens"] = sum(tokenizer_output["attention_mask"])
    return row


df = df.apply(tokenization, axis=1)

additional_tags = [
    "CI",
    "year",
    "oai_set_specs",
    
    "resumes.en",
    "topics.en",

    "resumes.fr",
    "topics.fr",
]

for col in additional_tags:
    df[col] = df[col].astype(str)

ds = Dataset.from_pandas(df.loc[:, additional_tags + ["input_ids", "attention_mask"]])

# ====================================================================================

ds = ds.select(range(10))

# ====================================================================================

embeddings_exporter = ExportEmbeddings(ds, model_name)
print("device : ", embeddings_exporter.device)
embeddings_exporter.routine(2, additional_tags, f"./embeddings/{model_name.split('/')[-1]}-{language}")
