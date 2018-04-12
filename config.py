class TensorflowConfig(object):
    def __init__(self, 
                 max_sentence_len=None,
                 embed_len = None,
                 embed_file = None,
                 vocab_processor = None,
                 tain_batch_size = None,
                 validation_batch_size = None,
                 train_files = None,  # list of strings
                 validation_files = None  # list of strings
                 num_epochs = None
                 checkpoint_file = None  # string pointing to a checkpoint file
                 vocabulary = None # string pointing to a json file
                 
        MAX_SENTENCE_LEN = max_sentence_len
        EMBED_LEN = embed_len
        EMBED_FILE = embed_file
        VOCAB_PROCESSOR = vocab_processor
        TRAIN_BATCH_SIZE = train_batch_size
        VALIDATION_BATCH_SIZE = validation_batch_size
        TRAIN_FILES = train_files
        VALIDATION_FILES = validation_files
        NUM_EPOCHS = num_epochs
        CHECKPOINT_FILE = checkpoint_file
        VOCABULARY = vocabulary
