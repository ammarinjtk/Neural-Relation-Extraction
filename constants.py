WORD_EMBEDDING_DIM = 200
POS_EMBEDDING_DIM = 100
DISTANCE_EMBEDDING_DIM = 300
DEPENDENCY_EMBEDDING_DIM = 300
UNKNOWN_WORD = '_UNK_'
PADDING_WORD = '_PAD_'

ELMO_EMBEDDING_DIM = 400

BERT_FEATURE_DIM = 768

# SEED = 7408
SEED_LIST = [166977, 755352, 820719, 585845, 312510, 250906, 675969, 647216, 290780, 692033, 439708, 195900, 614636, 240825,
             545617, 651802, 239387, 489995, 910463, 427810, 920316, 730426, 979591, 618602, 811533, 194063, 689795, 773546, 795116, 863132]

DATA_DIR_PATH = "./data"

W2V_MODEL_PATH = "/hdd/ammarinjtk/wikipedia-pubmed-and-PMC-w2v.bin"

ELMO_MODEL_PATH = "/hdd/ammarinjtk/ELMO_model/revised_bacteria_pubmed/headentity_finetune_weights.hdf5"
ELMO_OPTIONS_PATH = "/hdd/ammarinjtk/ELMO_model/revised_bacteria_pubmed/options.json"

BERT_FEATURES_PATH = "/hdd/ammarinjtk/BERT_features/synonym_revised_headentity/finetune_bert.jsonl"
BERT_TKN_PATH = "/hdd/ammarinjtk/BERT_features/synonym_revised_headentity/tkn.txt"

OUTPUT_DIR_PATH = "/hdd/ammarinjtk/example_predictions/our"
