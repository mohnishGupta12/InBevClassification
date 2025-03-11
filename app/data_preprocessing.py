file_path = "C:/pythonproject/classification/InBevClassification/text_classification_ecomm/ecommerceDataset.csv"

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from typing import Dict , Tuple , List
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import logging
import nltk
from nltk.corpus import stopwords
import config  # Import the config file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

class DataProcessor:
    """
    A class to handle data loading, preprocessing, and balancing for text classification tasks.
    """
    def __init__(self) -> None:
        """
        Initializes the DataProcessor with configuration settings.
        """
        self.file_path: str = config.DATA_CONFIG["file_path"]
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(config.MODEL_CONFIG["pretrained_model"])
        self.batch_size: int = config.TRAINING_CONFIG["batch_size"]
        self.max_length: int = config.DATA_CONFIG["max_length"]
        self.test_size: float = config.DATA_CONFIG["train_test_split"]

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset from a CSV file.
        
        Returns:
            pd.DataFrame: The loaded dataset containing category labels and text.
        """
        df = pd.read_csv(self.file_path, header=None, names=["category", "text"])
        return df

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Preprocesses the dataset by tokenizing text and encoding labels.
        
        Args:
            df (pd.DataFrame): The input dataframe containing text and labels.
        
        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: Tokenized inputs and encoded labels.
        """
        df["text"] = df["text"].astype(str).fillna("")
        
        inputs = self.tokenizer(
            df["text"].tolist(),
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = torch.tensor(df['category'].astype('category').cat.codes, dtype=torch.long)

        return inputs, labels

    def balance_data(self, inputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Balances the dataset using RandomOverSampler to handle class imbalances.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Tokenized inputs with input IDs and attention masks.
            labels (torch.Tensor): Corresponding labels.
        
        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: Resampled inputs and labels.
        """
        ros = RandomOverSampler()
        input_ids_resampled, labels_resampled = ros.fit_resample(inputs["input_ids"], labels)
        attention_mask_resampled, _ = ros.fit_resample(inputs["attention_mask"], labels)
        
        inputs_resampled = {
            "input_ids": torch.tensor(input_ids_resampled),
            "attention_mask": torch.tensor(attention_mask_resampled),
        }
        labels_resampled = torch.tensor(labels_resampled)
        return inputs_resampled, labels_resampled

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[str]]:
        """
        Prepares data loaders for training and validation.
        
        Returns:
            Tuple[DataLoader, DataLoader, List[str]]: Training data loader, validation data loader, and category labels.
        """
        df = self.load_data()
        logger.info(f"Original class distribution: {Counter(df['category'])}")

        train_df, val_df = train_test_split(df, test_size=self.test_size, stratify=df["category"])
        train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

        train_inputs, train_labels = self.preprocess_data(train_df)
        val_inputs, val_labels = self.preprocess_data(val_df)

        train_inputs, train_labels = self.balance_data(train_inputs, train_labels)
        logger.info(f"Resampled class distribution: {Counter(train_labels.tolist())}")

        train_data = torch.utils.data.TensorDataset(
            train_inputs["input_ids"], train_inputs["attention_mask"], train_labels
        )
        val_data = torch.utils.data.TensorDataset(
            val_inputs["input_ids"], val_inputs["attention_mask"], val_labels
        )

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size)

        return train_loader, val_loader, df["category"].astype("category").cat.categories.tolist()
