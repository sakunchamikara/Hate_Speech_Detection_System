import pandas as pd
from datasets import Dataset
from training.training_preprocessor import TrainingPreprocessor

class DatasetLoader:
    def __init__(self, train_path="data/SOLD_train.tsv", test_path="data/SOLD_test.tsv", adhoc=False):
        self.train_path = train_path
        self.test_path = test_path
        self.adhoc = adhoc
        self.preprocessor = TrainingPreprocessor(adhoc=adhoc)

    def load_dataframe(self):
        train_df = pd.read_csv(self.train_path, sep="\t", encoding="utf-8")
        test_df = pd.read_csv(self.test_path, sep="\t", encoding="utf-8")
        return train_df, test_df

    def preprocess_dataframe(self, df):
        df["clean_text"] = df["text"].apply(self.preprocessor.clean)
        df["hate"] = df["label"].apply(lambda x: 1 if x == "OFF" else 0)
        df = df[["clean_text", "hate"]].rename(columns={"clean_text": "text", "hate": "label"})
        return df

    def convert_to_dataset(self, df):
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(self.preprocessor.tokenize, batched=True)
        dataset.set_format("torch")
        return dataset

    def load(self):
        train_df, test_df = self.load_dataframe()
        train_df = self.preprocess_dataframe(train_df)
        test_df = self.preprocess_dataframe(test_df)
        train_ds = self.convert_to_dataset(train_df)
        test_ds = self.convert_to_dataset(test_df)
        return train_ds, test_ds