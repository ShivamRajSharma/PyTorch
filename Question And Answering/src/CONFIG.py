import tokenizers

ROBERTA_PATH = 'input/'
tokenizers =  tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"{ROBERTA_PATH}/vocab.json",
    merges_file=f"{ROBERTA_PATH}/merges.txt",
    lowercase=True,
    add_prefix_space=True
)

ROBERTA_PATH = 'input/'

MODLE_PATH = 'model/model.bin'

Dropout = 0.2
Epochs = 10
Batch_Size = 2
Max_Len = 140