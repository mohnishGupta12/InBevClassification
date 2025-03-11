""""""
import torch
import logging
import numpy as np
import pandas as pd
from typing import Dict , Tuple , List
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from .data_preprocessing import DataProcessor
from config import TRAINING_CONFIG, MODEL_CONFIG, DATA_CONFIG 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTClassifier:
    """
    A BERT-based text classifier for sequence classification tasks.
    """
    def __init__(
        self, 
        num_labels: int, 
        model_name: str = 'bert-base-uncased', 
        max_length: int = 128, 
        lr: float = 2e-5, 
        epochs: int = 10, 
        patience: int = 3
    ) -> None:
        """
        Initializes the BERT classifier with given parameters.
        
        Args:
            num_labels (int): Number of output labels for classification.
            model_name (str, optional): Pretrained BERT model name. Defaults to 'bert-base-uncased'.
            max_length (int, optional): Maximum sequence length for tokenization. Defaults to 128.
            lr (float, optional): Learning rate for the optimizer. Defaults to 2e-5.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            patience (int, optional): Early stopping patience. Defaults to 3.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)
        self.max_length = max_length
        self.epochs = epochs
        self.lr = lr
        self.patience = patience

    def train_model(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader) -> BertForSequenceClassification:
        """
        Trains the BERT classifier.
        
        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
        
        Returns:
            BertForSequenceClassification: The trained BERT model.
        """
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                b_input_ids, b_input_mask, b_labels = [item.to(self.device) for item in batch]
                self.model.zero_grad()
                outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = total_loss / len(train_loader)
            logger.info(f'Epoch {epoch + 1}, Training loss: {avg_train_loss}')
            
            val_loss, val_accuracy = self.evaluate(val_loader)
            logger.info(f'Epoch {epoch + 1}, Validation loss: {val_loss}, Validation accuracy: {val_accuracy}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info('Early stopping triggered.')
                    break

        self.model.load_state_dict(torch.load('best_model.pth'))
        return self.model

    def evaluate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Evaluates the model on the validation set.
        
        Args:
            val_loader (DataLoader): Validation data loader.
        
        Returns:
            Tuple[float, float]: The average validation loss and accuracy.
        """
        self.model.eval()
        total_loss = 0
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                b_input_ids, b_input_mask, b_labels = [item.to(self.device) for item in batch]
                outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()
                logits = outputs.logits
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                true_labels.extend(b_labels.cpu().numpy())
        
        avg_val_loss = total_loss / len(val_loader)
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        logger.info(f'Validation Report:\n{classification_report(true_labels, predictions)}')
        return avg_val_loss, accuracy
