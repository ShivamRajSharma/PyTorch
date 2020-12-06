english_path = 'input/train.en'
german_path = 'input/train.de'

encoder_embed_dims = 250
encoder_hidden_dims = 250
encoder_num_layers = 1
encoder_dropout = 0.25

teacher_force_ratio = 0.5

decoder_embed_dims = 250
decoder_hidden_dims = 250
decoder_num_layers = 1
decoder_dropout = 0.25

LR = 1e-3
scheduler_thresh = 0.05
patience = 3
decay_factor=0.6

EPOCHS = 20
BATCH_SIZE = 32
MODEL_PATH = '../model/model.bin'
WARMUP_STEPS = 0
