import transformers 

max_len = 200
Epochs = 2
Batch_Size = 8

BERT_Path = '../input/'
Model_Path= '../model/bert_sentiment.bin'
Tokenizer = transformers.BertTokenizer.from_pretrained(BERT_Path, do_lower_case=True)

pad_idx = Tokenizer.encode('[PAD]')[1]
