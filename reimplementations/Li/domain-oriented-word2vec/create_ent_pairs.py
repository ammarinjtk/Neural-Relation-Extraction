from imports import *

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


tkns = []
tkn_file_paths = [filename for filename in glob.glob(
    os.path.join(PICKLE_PATH, "*.pickle")) if 'tkns' in filename]
tkn_count = 0

for tkn_file_path in tqdm(tkn_file_paths, desc="Tkn_file_paths loop"):
    with open(tkn_file_path, 'rb') as f:
        tkns = pickle.load(f)

    idx_pairs = []

    # for each sentence
    for tkn in tqdm(tkns, desc="Sentences loop"):

        indices = [
            {'word': word2idx[word['word']],
             'stem': word2idx[word['stem']],
             'chunk': word2idx[word['chunk']],
             'entity': word2idx[word['entity']]} for word in tkn
        ]

        # for each word, threated as center word
        for center_word_pos in range(len(indices)):
            # create huffman nodes from root to center word
            center_word = idx2word[indices[center_word_pos]['word']]
            center_word_idx = indices[center_word_pos]
            center_code = word2codec[idx2word[indices[center_word_pos]['word']]]
            node_codes = []
            for i in range(len(center_code)):
                if center_code[:i] in nodes:
                    node_codes.append(center_code[:i])

            # for each window position
            unique_window_units = []
            for w in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
                context_word_pos = center_word_pos + w
                # make soure not jump out sentence
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                for key, value in context_word_idx.items():
                    if value not in unique_window_units:
                        unique_window_units.append(value)

            idx_pairs.append({
                'center': {'center_idx': center_word_idx,
                           'center_code':  {
                               'word': center_code,
                               'stem': word2codec[idx2word[indices[center_word_pos]['stem']]],
                               'chunk': word2codec[idx2word[indices[center_word_pos]['chunk']]],
                               'entity': word2codec[idx2word[indices[center_word_pos]['entity']]]
                           }},
                'context_word_indices': [word_idx for word_idx in unique_window_units],
                'nodes': [{'node_idx': node2idx[node_code], 'node_code': node_code} for node_code in node_codes]}
            )

    with open(os.path.join(PICKLE_PATH, f"idx_pairs_{tkn_count}.pickle"), 'wb') as f:
        pickle.dump(idx_pairs, f)

    tkn_count += 1
