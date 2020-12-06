import tokenizers

ROBERTA_PATH = 'input/'
tokenizers =  tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"{ROBERTA_PATH}/vocab.json",
    merges_file=f"{ROBERTA_PATH}/merges.txt",
    lowercase=True,
    add_prefix_space=True
)

ROBERTA_PATH = 'input/'

MODEL_PATH = 'model/model.bin'

LR = 4e-5
Dropout = 0.1
Epochs = 3
Batch_Size = 32
Max_Len = 192