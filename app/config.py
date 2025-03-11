TRAINING_CONFIG = {
    "epochs": 10,
    "learning_rate": 2e-5,
    "patience": 3,
    "batch_size": 16
}

MODEL_CONFIG = {
    "pretrained_model": "bert-base-uncased",
    "num_labels": 2
}

DATA_CONFIG = {
    "max_length": 128,
    "train_test_split": 0.2,
    "file_path": "app/text_classification_ecomm/ecommerceDataset.csv"
}