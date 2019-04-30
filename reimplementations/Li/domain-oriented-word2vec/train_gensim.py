from imports import *
from constants import *

ENTITY_TYPES = ['cell_type', 'protein', 'DNA', 'cell_line', 'RNA']
WINDOW_SIZE = 5
EMBEDDING_DIM = 50

with open(os.path.join(PICKLE_PATH, "word2idx.pickle"), 'rb') as f:
    word2idx = pickle.load(f)

with open(os.path.join(PICKLE_PATH, "idx2word.pickle"), 'rb') as f:
    idx2word = pickle.load(f)

with open(os.path.join(PICKLE_PATH, "word2codec.pickle"), 'rb') as f:
    word2codec = pickle.load(f)

with open(os.path.join(PICKLE_PATH, "node2idx.pickle"), 'rb') as f:
    node2idx = pickle.load(f)

with open(os.path.join(PICKLE_PATH, "idx2node.pickle"), 'rb') as f:
    idx2node = pickle.load(f)

with open(os.path.join(PICKLE_PATH, "nodes.pickle"), 'rb') as f:
    nodes = pickle.load(f)

idx_pair_file_paths = [filename for filename in glob.glob(
    os.path.join(PICKLE_PATH, "*.pickle")) if 'idx_pairs' in filename]

with open(idx_pair_file_paths[0], 'rb') as f:
    idx_pairs = pickle.load(f)

sentences = []
for idx_pair in tqdm(idx_pairs, desc="Idx_pair loop"):
    for word_type in ['word', 'stem', 'chunk', 'entity']:
        center_idx = idx_pair['center']['center_idx'][word_type]

        for context_idx in idx_pair['context_word_indices']:
            sentences.append([idx2word[center_idx], idx2word[context_idx]])

model = gensim.models.Word2Vec(sentences, size=WORD_EMBEDDING_DIM, sg=1, hs=1,
                               window=1, compute_loss=True, workers=4, iter=EPOCH, alpha=0.025)

for idx in tqdm(range(1, 3), desc="Idx_pair_file loop"):
    with open(idx_pair_file_paths[idx], 'rb') as f:
        idx_pairs = pickle.load(f)
    sentences = []
    for idx_pair in tqdm(idx_pairs, desc="Idx_pair loop"):
        for word_type in ['word', 'stem', 'chunk', 'entity']:
            center_idx = idx_pair['center']['center_idx'][word_type]

            for context_idx in idx_pair['context_word_indices']:
                sentences.append([idx2word[center_idx], idx2word[context_idx]])
    model.build_vocab(sentences, update=True)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)

model.save(os.path.join(LI_W2V_OUTPUT_DIR_PATH, "li_w2v.model"))
