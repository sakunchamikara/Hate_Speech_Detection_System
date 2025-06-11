class Config:
    MODEL_NAME = "xlm-roberta-base"
    MAX_LENGTH = 128
    NUM_LABELS = 2
    NUM_EPOCHS = 3
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    USE_ADHOC_TRANSLITERATION = True