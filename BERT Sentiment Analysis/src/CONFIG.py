import transformers 

max_len = 200
Epochs = 2
Batch_Size = 8
BERT_Path = 'input/'
Model_Path= '../model/bert_sentiment.bin'
Tokenizer = transformers.BertTokenizer.from_pretrained(BERT_Path, do_lower_case=True)

# sentence = '[PAD]'

# inputs = Tokenizer.encode(sentence)
# print(inputs)

# ids = inputs['ids']
# mask = inputs['mask']
# token_type_ids = inputs['token_type_ids']
# print(ids)

