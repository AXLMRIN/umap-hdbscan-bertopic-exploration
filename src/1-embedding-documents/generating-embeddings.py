from datasets import Dataset
import pandas as pd
from transformers import AutoConfig
from sentence_transformers import SentenceTransformer
from torch.cuda import is_available as cuda_available

from ExportEmbeddingsClass import clean

language = "fr"
# ==============================================================================
df = pd.read_csv("./data/theses-to-embed.csv").sample(frac = 1)
text_col = f"resumes.{language}"
# model_name = "Qwen/Qwen3-Embedding-0.6B" # multilingual 
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # multilingual
# model_name = "Alibaba-NLP/gte-multilingual-base" # multilingual 
# model_name = "codefuse-ai/F2LLM-0.6B" # english 
# model_name = "Sahajtomar/french_semantic" # french
# model_name = "camembert/camembert-base" # fr
# model_name = "almanach/camembertav2-base" # fr

tags = [
    "CI",
    "oai_set_specs",
    "resumes.fr",
    "topics.fr",
    "resumes.en",
    "topics.en",
    "year",
]

for col in tags: df[col] = df[col].astype(str)

ds = Dataset.from_pandas(df.loc[:,tags])

# ==============================================================================

ds = ds.select(range(10))

# ==============================================================================
device = "cuda" if cuda_available() else "cpu"
print(f"Device : {device}")
sbert_model = SentenceTransformer(
    model_name, 
    device = device, 
    trust_remote_code = True
)
sbert_model.max_seq_length = \
    AutoConfig.from_pretrained(model_name).max_position_embeddings

try : 
    embeddings = (
        sbert_model
        .encode(
            ds["resumes.fr"], 
            device=str(device), 
            normalize_embeddings=True, 
            show_progress_bar=True
        )
    )
    print(embeddings.shape)
    ds = ds.add_column(
        "embedding", 
        [
            embeddings[i,:].reshape(-1,) 
            for i in range(embeddings.shape[0])
        ]
    )
    ds.save_to_disk("temp-test")
except Exception as e:
    print(e)
finally:
    del sbert_model, ds
    clean()
