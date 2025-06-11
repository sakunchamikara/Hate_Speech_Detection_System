import pandas as pd
from datasets import Dataset
from training.preprocessor import clean_and_transliterate, tokenize_function

def load_dataset(train_path="data/SOLD_train.tsv", test_path="data/SOLD_test.tsv"):

    # Load with pandas
    train_df = pd.read_csv(train_path, sep="\t", encoding="utf-8")
    test_df = pd.read_csv(test_path, sep="\t", encoding="utf-8")

    return train_df, test_df

def preprocess_dataframe(df):
    
    df["clean_text"] = df["text"].apply(clean_and_transliterate)
    df["hate"] = df["label"].apply(lambda x: 1 if x == "OFF" else 0)
    df = df[["clean_text", "hate"]].rename(columns={"clean_text": "text", "hate": "label"})
    return df

def convert_to_dataset(df):
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format("torch")
    return dataset

def load_dataset_pipeline(train_path="data/SOLD_train.tsv", test_path="data/SOLD_test.tsv"):

    train_df, test_df = load_dataset(train_path, test_path)
    train_df = preprocess_dataframe(train_df)
    test_df = preprocess_dataframe(test_df)
    train_ds = convert_to_dataset(train_df)
    test_ds = convert_to_dataset(test_df)
    return train_ds, test_ds
