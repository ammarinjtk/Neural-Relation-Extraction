from imports import *


def build_vocab(words, n_words):
    """Process raw inputs into a ."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return count, dictionary, reversed_dictionary


def save_vocab(save_path, rev_vocab):
    with open(os.path.join(save_path, "vocab.txt"), "w") as f:
        for i in range(len(rev_vocab)):
            vocab_word = rev_vocab[i]
            f.write("%s\n" % (vocab_word))


ENTITY_TYPES = ['cell_type', 'protein', 'DNA', 'cell_line', 'RNA']
WINDOW_SIZE = 5
EMBEDDING_DIM = 50

words = []
unique_chunks = []
unique_entity = []

tkn_file_paths = [filename for filename in glob.glob(
    os.path.join(PICKLE_PATH, "*.pickle")) if 'tkns' in filename]
tkn_count = 0

for tkn_file_path in tqdm(tkn_file_paths):
    with open(tkn_file_path, 'rb') as f:
        for sentence in tqdm(pickle.load(f)):
            for w in sentence:
                words.append(w['word'])
                words.append(w['stem'])

                if w['chunk'] not in unique_chunks:
                    unique_chunks.append(w['chunk'])
                    words.append(w['chunk'])
                if w['entity'] not in unique_entity:
                    unique_entity.append(w['entity'])
                    words.append(w['entity'])

word_cnt, word2idx, idx2word = build_vocab(words, len(words))

with open(os.path.join(PICKLE_PATH, "word_cnt.pickle"), 'wb') as f:
    pickle.dump(word_cnt, f)

with open(os.path.join(PICKLE_PATH, "word2idx.pickle"), 'wb') as f:
    pickle.dump(word2idx, f)

with open(os.path.join(PICKLE_PATH, "idx2word.pickle"), 'wb') as f:
    pickle.dump(idx2word, f)

codec = HuffmanCodec.from_frequencies(dict(word_cnt))
word2codec = {}
codec_table = codec.get_code_table()
for key, value in codec_table.items():
    word2codec[key] = bin(value[1])[2:].rjust(value[0], '0')

with open(os.path.join(PICKLE_PATH, "word2codec.pickle"), 'wb') as f:
    pickle.dump(word2codec, f)

prefixes = []
for key, value in word2codec.items():
    prefixes.append(value)
prefixes.sort(key=lambda s: len(s))

nodes = set()
for prefix in prefixes:
    for i in range(len(prefix)):
        if prefix[:i] != "":
            nodes.add(prefix[:i])

nodes = list(nodes)
nodes.sort(key=lambda s: len(s))

node2idx = {}
for idx, node in enumerate(nodes):
    node2idx[node] = idx

idx2node = dict(zip(node2idx.values(), node2idx.keys()))

with open(os.path.join(PICKLE_PATH, "node2idx.pickle"), 'wb') as f:
    pickle.dump(node2idx, f)

with open(os.path.join(PICKLE_PATH, "idx2node.pickle"), 'wb') as f:
    pickle.dump(idx2node, f)

with open(os.path.join(PICKLE_PATH, "nodes.pickle"), 'wb') as f:
    pickle.dump(nodes, f)
