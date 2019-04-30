WORD_EMBEDDING_DIM = 200
POS_EMBEDDING_DIM = 100
DISTANCE_EMBEDDING_DIM = 300
DEPENDENCY_EMBEDDING_DIM = 300
UNKNOWN_WORD = '_UNK_'
PADDING_WORD = '_PAD_'

ELMO_EMBEDDING_DIM = 400

BERT_FEATURE_DIM = 768

# SEED = 239387

SEED_LIST = [345, 166977, 239387, 240825, 250906, 290780, 312510, 439708, 489995, 
             545617, 585845, 614636, 675969, 820719, 920316, 1187617, 1484458, 3835082, 
             5064379, 5647183, 5694250, 6333898, 6797546, 7144728, 7461780, 7696045, 8468732, 
             8730848, 8842617, 9975400]

DATA_DIR_PATH = "./data"

W2V_MODEL_PATH = "/home/mind/ammarinjtk/wikipedia-pubmed-and-PMC-w2v.bin"

ELMO_MODEL_PATH = "/home/mind/ammarinjtk/Neural-Relation-Extraction/entitykeyword_finetune_weights.hdf5"
ELMO_OPTIONS_PATH = "/home/mind/ammarinjtk/Neural-Relation-Extraction/options.json"

BERT_FEATURES_PATH = "/home/mind/ammarinjtk/Neural-Relation-Extraction/finetuned_bert.jsonl"

OUTPUT_DIR_PATH = "./our_predictions"
