"""
Data cleaning of theses-soutenues.csv (in folder data). 
Steps : 
    - select thesis published between 2010 and 2022, remove all rows where the 
      oai_set_specs, resumes.fr and resumes.en are NOT strings
    - aggregate the topics (topics.fr and topics.en)
    - make sure the resumes provided are written in the right language, if both
      resumes are provided but inserted in the wrong column, swap the values  
    - retrieve all entries where the languages match.
    - add a custom index
"""
from time import time
from fast_langdetect import detect_language as detect
import numpy as np
import pandas as pd
pd.options.mode.copy_on_write = True  

columns = [
    "date_soutenance",
    "discipline",
    "oai_set_specs",
    "titres.en",
    "titres.fr",
    "resumes.fr",
    "resumes.en",
    "sujets.en",
    "sujets.fr",
    # 'resumess.autre.0',
    # 'resumes.autre.1',
    # 'resumes.autre.2',
    # 'resumes.autre.3',
    # 'resumes.autre.4',
    # 'resumes.autre.5',
    "sujets.autre.0",
    "sujets.autre.1",
    "sujets.autre.2",
    "sujets.autre.3",
    "sujets.autre.4",
    "sujets.autre.5",
    "sujets.autre.6",
    "sujets.autre.7",
    "sujets_rameau.0",
    "sujets_rameau.1",
    "sujets_rameau.10",
    "sujets_rameau.11",
    "sujets_rameau.12",
    "sujets_rameau.13",
    "sujets_rameau.14",
    "sujets_rameau.15",
    "sujets_rameau.16",
    "sujets_rameau.17",
    "sujets_rameau.18",
    "sujets_rameau.19",
    "sujets_rameau.2",
    "sujets_rameau.20",
    "sujets_rameau.21",
    "sujets_rameau.22",
    "sujets_rameau.23",
    "sujets_rameau.24",
    "sujets_rameau.25",
    "sujets_rameau.26",
    "sujets_rameau.27",
    "sujets_rameau.28",
    "sujets_rameau.29",
    "sujets_rameau.3",
    "sujets_rameau.30",
    "sujets_rameau.31",
    "sujets_rameau.32",
    "sujets_rameau.33",
    "sujets_rameau.34",
    "sujets_rameau.35",
    "sujets_rameau.36",
    "sujets_rameau.37",
    "sujets_rameau.38",
    "sujets_rameau.39",
    "sujets_rameau.4",
    "sujets_rameau.40",
    "sujets_rameau.41",
    "sujets_rameau.42",
    "sujets_rameau.43",
    "sujets_rameau.44",
    "sujets_rameau.45",
    "sujets_rameau.46",
    "sujets_rameau.47",
    "sujets_rameau.48",
    "sujets_rameau.49",
    "sujets_rameau.5",
    "sujets_rameau.50",
    "sujets_rameau.51",
    "sujets_rameau.52",
    "sujets_rameau.53",
    "sujets_rameau.54",
    "sujets_rameau.6",
    "sujets_rameau.7",
    "sujets_rameau.8",
    "sujets_rameau.9",
]
start = time()
df_raw = pd.read_csv("./data/theses-soutenues.csv", usecols=columns)
print(f"# df loaded (nrows = {len(df_raw)} - {time()-start:.0f} s)")

# First select rows to reduce the size of the dataframe ========================
def get_year(date : str):
    try : 
        return pd.Timestamp(date).year
    except Exception:
        return np.nan

start = time()
df_raw.loc[:,"year"] = df_raw.loc[:,"date_soutenance"].apply(get_year)
# Select thesis published between 2010 and 2022
def is_string(el): return isinstance(el, str)
years_to_keep = np.logical_and.reduce([
    df_raw["year"] >= 2010, 
    df_raw["year"] <= 2022,
    df_raw["oai_set_specs"].apply(is_string),   
    df_raw["resumes.en"].apply(is_string),   
    df_raw["resumes.fr"].apply(is_string),   
])
df = df_raw.loc[years_to_keep, :]
print((f"# only keep thesis published between 2010 and 2022 and inputs where"
       f" the oai_set_specs and resumes are the right type (string) "
       f"(nrows = {len(df)};"
       f" {100 * len(df)/len(df_raw):.0f} % - {time()-start:.0f} s)"))

# Aggregation of topics and language checking ==================================
def aggregate_sujets_rameau(row: dict[str]):
    sujets_en = []
    sujets_fr = []
    for col_name in row.index:
        if col_name.startswith("sujets_rameau"):
            if isinstance(row[col_name], str):
                language = detect(row[col_name])
                if language == "FR":
                    sujets_fr += [row[col_name]]
                elif language == "EN":
                    sujets_en += [row[col_name]]
    row["topics.fr"] = "||".join(sujets_fr)
    row["topics.en"] = "||".join(sujets_en)
    return row

start = time()
df = df.apply(aggregate_sujets_rameau, axis=1)
print(f"# Aggregation of the topics ({time() - start:.0f} s)")

cols = [
    "year",
    "oai_set_specs",

    "titres.en",
    "resumes.en",
    "topics.en",
    
    "titres.fr",
    "resumes.fr",
    "topics.fr",
]
df = df.loc[:, cols]


def check_language_resumes(row: dict):
    row["lang_res.en"] = np.nan
    row["lang_res.fr"] = np.nan
    # check language for english column
    if isinstance(row["resumes.en"], str):
        if len(row["resumes.en"]) > 0:
            row["lang_res.en"] = detect(row["resumes.en"])
    # check language for french column
    if isinstance(row["resumes.fr"], str):
        if len(row["resumes.fr"]) > 0:
            row["lang_res.fr"] = detect(row["resumes.fr"])

    return row


start = time()
df = df.apply(check_language_resumes, axis=1)
print(f"# resumes' language checked ({time() - start:.0f} s)")

def swap_languages(row: dict):
    row_copy = row.copy()
    if (row["lang_res.en"] == "FR") and (row["lang_res.fr"] == "EN"):
        row_copy["resumes.en"] = row["resumes.fr"]
        row_copy["resumes.fr"] = row["resumes.en"]
        row_copy["lang_res.fr"] = "FR"
        row_copy["lang_res.en"] = "EN"
        row_copy["swapped"] = True
    return row_copy

start = time()
df = df.apply(swap_languages, axis=1)
print(f"# resumes swapped ({time() - start:.0f} s)")

# Save file as parquet ==========================================================
rows_to_keep = np.logical_and(
    (df["lang_res.en"] == "EN"),
    (df["lang_res.fr"] == "FR")
)
cols_to_keep = [
    "year",
    "oai_set_specs",
    
    "titres.en",
    "resumes.en",
    "lang_res.en",
    "topics.en",
    
    "titres.fr",
    "resumes.fr",
    "lang_res.fr",
    "topics.fr",
    
    "swapped",
]
df_final = df.loc[rows_to_keep, cols_to_keep]

# Create a custom index
df_final.insert(0,"CI",[f"CI-{i}" for i in range(len(df_final))])

df_final.to_csv("./data/theses-soutenues-clean-with-index.csv", index=False)
df_final.to_parquet("./data/theses-soutenues-clean-with-index.parquet", index=False)
print((f"# file saved (nrows = {len(df_final)}; "
       f"{int(100 * len(df_final)/len(df))} %)"))
