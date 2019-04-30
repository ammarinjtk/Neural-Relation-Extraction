from imports import *
from constants import *

def is_number(word):
    try:
        float(word)
    except ValueError:
        return False
    else:
        return True

def preprocess(file_path):
    dataloaders = {}
    pos_examples = 0
    neg_examples = 0
    
    removal_relations = {
        'train': {'inter_sentence': 0, 'no_shortest': 0},
        'dev': {'inter_sentence': 0, 'no_shortest': 0}
    }

    for phase in ['train', 'dev', 'test']:
        file = minidom.parse(f"{file_path}BioNLP-ST-2016_BB-event_{phase}.xml")
        docs = file.getElementsByTagName("document")
        trn_data = []
        trn_label = []
        for doc in docs:
            document_idx = doc.getAttribute("origId")
            sentences = doc.getElementsByTagName("sentence")
            for sentence in sentences:
                
                sentence_idx = sentence.getAttribute("id")
                entities = sentence.getElementsByTagName("entity")
                interactions = sentence.getElementsByTagName("interaction")
                analyses = sentence.getElementsByTagName("analyses")[0]
                tokens = analyses.getElementsByTagName("tokenization")[0].getElementsByTagName("token")
                dependencies = analyses.getElementsByTagName("parse")[0].getElementsByTagName("dependency")

                entity_type_dict = {
                    'Habitat': [],
                    'Bacteria': [],
                    'Geographical': []
                }
                entity_pairs = []
                entity_to_charoffset = {}
                entity_idx_to_type = {}
                entity_idx_to_origId = {}
                entity_idx_to_text = {}

                for idx, entity in enumerate(entities):
                    entity_type = entity.getAttribute("type")
                    entity_origId = entity.getAttribute("origId")
                    entity_idx = entity.getAttribute("id")
                    entity_idx_to_type[entity_idx] = entity_type
                    entity_idx_to_origId[entity_idx] = entity_origId
                    entity_to_charoffset[entity_idx] = entity.getAttribute("headOffset")
                    entity_idx_to_text[entity_idx] = entity.getAttribute("text")
                    
                    if entity_type in entity_type_dict:
                        entity_type_dict[entity_type].append(entity_idx)
                    
                for bacteria in entity_type_dict['Bacteria']:
                    for location in entity_type_dict['Habitat']+entity_type_dict['Geographical']:
                        # Exclude cross-sentence relations
                        try:
                            first_sentence = re.match(r'TEES.d\d+.(s\d+).e\d+', bacteria).group(1)
                            second_sentence = re.match(r'TEES.d\d+.(s\d+).e\d+', location).group(1)
                        except AttributeError:
                            continue
                        else:
                            if first_sentence == second_sentence:
                                entity_pairs.append((bacteria, location))

                label_pairs = []
                tmp_label_pairs = []
                for interaction in interactions:
                    tmp_label_pairs.append((interaction.getAttribute("e1"), interaction.getAttribute("e2")))
                    
                    if phase in ['train', 'dev']:
                        if not (interaction.getAttribute("e1"), interaction.getAttribute("e2")) in entity_pairs:
                            removal_relations[phase]['inter_sentence'] += 1

                for entity_pair in entity_pairs:
                    if entity_pair in tmp_label_pairs:
                        label_pairs.append(1)
                        pos_examples += 1
                    else:
                        label_pairs.append(0)
                        neg_examples += 1

                token_list = []
                for token in tokens:
                    tmp_token_dict = {}
                    tmp_token_dict['id'] = token.getAttribute("id")
                    tmp_token_dict['text'] = token.getAttribute("text")
                    tmp_token_dict['charOffset'] = token.getAttribute("charOffset")
                    tmp_token_dict['POS'] = token.getAttribute("POS")

                    token_list.append(tmp_token_dict)

                dep_list = []
                edges = []
                for dependency in dependencies:

                    edges.append((dependency.getAttribute("t1"), dependency.getAttribute("t2")))

                    dep_list.append({
                        'id': dependency.getAttribute("id"),
                        'path': (dependency.getAttribute("t1"), dependency.getAttribute("t2")),
                        'type': dependency.getAttribute("type")
                    })

                start_token = ""
                end_token = ""
                for token in tokens:
                    tmp_token_dict = {}
                    tmp_token_dict['id'] = token.getAttribute("id")
                    tmp_token_dict['text'] = token.getAttribute("text")
                    tmp_token_dict['charOffset'] = token.getAttribute("charOffset")
                    tmp_token_dict['POS'] = token.getAttribute("POS")

                # Contains entity_1 and entity_2
                start_end_list = []
                for (start_entity, end_entity) in entity_pairs:
                    start_token = ""
                    end_token = ""
                    for token in token_list:
                        if token['charOffset'] == entity_to_charoffset[start_entity]:
                            start_token = token['id']
                        if token['charOffset'] == entity_to_charoffset[end_entity]:
                            end_token = token['id']
                    start_end_list.append([(start_token, end_token), (start_entity, end_entity)])

                shortest_input_list = []
                semifull_input_list = []
                full_input_list = []

                for entity_idx, ((start_token, end_token), (start_entity, end_entity)) in enumerate(start_end_list):
                    
                    graph = nx.Graph(edges)
                    try:
                        shortest_path = nx.shortest_path(graph, source=start_token, target=end_token)
                    except nx.NetworkXNoPath:
                        # print("-- NetworkXNoPath --")
                        shortest_path = None
                    except nx.NodeNotFound:
                        # print("-- NodeNotFound --")
                        shortest_path = None
                    else:
                        if len(shortest_path) == 1:
                            # print("-- ShortestPathOnlyOneItem (entity pair is the same words) --")
                            shortest_path = None

                    # Find entity_1 and entity_2 idx
                    entity_1_idx = 0
                    entity_2_idx = 0
                    for idx, token in enumerate(token_list): 
                        if token['id'] == start_token:
                            entity_1_idx = idx
                        if token['id'] == end_token: 
                            entity_2_idx = idx

                    shortest_token_list = []
                    shortest_pos_list = []
                    shortest_dist1_list = []
                    shortest_dist2_list = []
                    shortest_dep_list = []

                    full_token_list = []
                    full_pos_list = []
                    full_dist1_list = []
                    full_dist2_list = []
                    full_dep_list = []
                    
                    if shortest_path:
                        # print("### Shortest! ###")
                        for token_idx in shortest_path:
                            for idx, token in enumerate(token_list):
                                if token['id'] == token_idx:
                                    if token['id'] == start_token:
                                        shortest_token_list.append('entity_1')

                                            
                                    elif token['id'] == end_token:
                                        shortest_token_list.append('entity_2')

                                    else:
                                        if is_number(token['text']):
                                            shortest_token_list.append("0")
                                        else:
                                            shortest_token_list.append(token['text'].lower())
                                    shortest_pos_list.append(token['POS'])
                                    shortest_dist1_list.append(entity_1_idx-idx)
                                    shortest_dist2_list.append(entity_2_idx-idx)
                                    
                        for idx, _ in enumerate(shortest_path):
                            try:
                                path = (shortest_path[idx], shortest_path[idx+1])
                                rev_path = (shortest_path[idx+1], shortest_path[idx])
                            except IndexError:
                                break
                            else:
                                for dep in dep_list:
                                    if dep['path'] == path or dep['path'] == rev_path:
                                        shortest_dep_list.append(dep['type'])

                    else:
                        # print("### Shortest not found! ###")
                        if phase in ['train', 'dev']:
                            removal_relations[phase]['no_shortest'] += 1
                        continue
                    
                    shortest_input_list.append({
                        'shortest_token': shortest_token_list,
                        'shortest_pos': shortest_pos_list,
                        'shortest_dist1': shortest_dist1_list,
                        'shortest_dist2': shortest_dist2_list,
                        'shortest_dep': shortest_dep_list
                    })

                    # Full path
                    for idx, token in enumerate(token_list):
                        if token['id'] == start_token:
                            full_token_list.append('entity_1')
                        elif token['id'] == end_token:
                            full_token_list.append('entity_2')
                        else:
                            if is_number(token['text']):
                                # print(f"Full-sentence, replace {token['text']} with {'0'}")
                                full_token_list.append("0")
                            else:
                                full_token_list.append(token['text'].lower())
                        full_pos_list.append(token['POS'])
                        full_dist1_list.append(entity_1_idx-idx)
                        full_dist2_list.append(entity_2_idx-idx)
                    for idx, dep in enumerate(dep_list):
                            full_dep_list.append(dep['type'])

                    full_input_list.append({
                        'full_token': full_token_list,
                        'full_pos': full_pos_list,
                        'full_dist1': full_dist1_list,
                        'full_dist2': full_dist2_list,
                        'full_dep': full_dep_list
                    })
                                  

                for idx, shortest_input in enumerate(shortest_input_list):
                    trn_data.append({
                                'document_id': document_idx,
                                'sentence_id': sentence_idx,
                                'shortest_inputs': shortest_input,
                                'full_inputs': full_input_list[idx],
                                'entity_pair': entity_pairs[idx],
                                'label': label_pairs[idx],
                                'entity_idx_to_type': entity_idx_to_type,
                                'entity_idx_to_origId': entity_idx_to_origId,
                            })
                    trn_label.append(label_pairs[idx])

        dataloaders[phase] = trn_data
    print("Load data successfully!")
    print(f"Removal relations for train: {removal_relations['train']['inter_sentence']} cross-sentence and {removal_relations['train']['no_shortest']} no-SDP")
    print(f"Removal relations for dev: {removal_relations['dev']['inter_sentence']} cross-sentence and {removal_relations['dev']['no_shortest']} no-SDP")                            
    return dataloaders


def prepare_sequence(sentence, to_ix, max_length, prepare_word=True, padding=False, get_lengths=False):
    words = []
    idxs = []
    for idx, word in enumerate(sentence):
        words.append(word)
        try:
            to_ix[word]
        except KeyError:
            if prepare_word and idx == 0:
                # Handle OOV bacteria mention which is always the first word if there is shortest path.
#                 print("{}, which is bacteria mention, is UNKNOWN".format(word))
                idxs.append(to_ix['bacteria'])
            else:
                idxs.append(to_ix[UNKNOWN_WORD])
        else:
            idxs.append(to_ix[word])
    if padding and len(sentence) < max_length:
        for idx in range(max_length-len(sentence)):
            words.append(PADDING_WORD)
            idxs.append(to_ix[PADDING_WORD])
        if len(idxs) != max_length:
            raise ValueError(f"words does not have length {len(words)} equals max_length {max_length} after padding.")
            
    if get_lengths:
        return words, torch.tensor(idxs, dtype=torch.long), torch.tensor(len(sentence), dtype=torch.long)
    else:
        return words, torch.tensor(idxs, dtype=torch.long)


def build_vocab(dataloaders, w2v_model, vocabs):
    
    word_to_ix = {}
    pos_to_ix = {}
    distance_to_ix = {}
    dependency_to_ix = {}
    char_to_ix = {}
    in_vocab_count = {
        'VOCAB': 0,
        'OOV': 0
    }
    
    word_to_ix[PADDING_WORD] = len(word_to_ix)
    pos_to_ix[PADDING_WORD] = len(pos_to_ix)
    distance_to_ix[PADDING_WORD] = len(distance_to_ix)
    dependency_to_ix[PADDING_WORD] = len(dependency_to_ix)
    char_to_ix[PADDING_WORD] = len(char_to_ix)
    
    for vocab, count in vocabs:
        word_to_ix[vocab] = len(word_to_ix)
    
    for input_dict in dataloaders['train']+dataloaders['dev']+dataloaders['test']:
        inputs = input_dict[f'full_inputs']
        for token in inputs[f'full_token']:
            if token not in word_to_ix:
                # print("OOV:", token)
                in_vocab_count['OOV'] += 1
            else:
                in_vocab_count['VOCAB'] += 1
                
            for char in token:
                if char not in char_to_ix:
                    char_to_ix[char] = len(char_to_ix)
        for pos in inputs[f'full_pos']:
            if pos not in pos_to_ix:
                pos_to_ix[pos] = len(pos_to_ix)
        for dep in inputs[f'full_dep']:
            if dep not in dependency_to_ix:
                dependency_to_ix[dep] = len(dependency_to_ix)
        for dist in inputs[f'full_dist1']+inputs[f'full_dist2']:
            if dist not in distance_to_ix:
                distance_to_ix[dist] = len(distance_to_ix)
    
    pos_to_ix[UNKNOWN_WORD] = len(pos_to_ix)
    distance_to_ix[UNKNOWN_WORD] = len(distance_to_ix)
    word_to_ix[UNKNOWN_WORD] = len(word_to_ix)
    dependency_to_ix[UNKNOWN_WORD] = len(dependency_to_ix)
    char_to_ix[UNKNOWN_WORD] = len(char_to_ix)

    return word_to_ix, pos_to_ix, distance_to_ix, dependency_to_ix, char_to_ix, in_vocab_count

def build_pretrain_embedding_matrix(w2v_model, word_to_ix, distance_to_ix, max_distance, dist_dim=DISTANCE_EMBEDDING_DIM):
    pretrained_embedding_matrix = []
    distance_pretrain_embedding_matrix = []
    
    # Random initialized padding and unk vectors
    rand_vect = np.random.randn(WORD_EMBEDDING_DIM)*0.01
    dist_rand_vect = np.random.randn(dist_dim)*0.01
    
    # For padding word
    pretrained_embedding_matrix.append(rand_vect)
    distance_pretrain_embedding_matrix.append(dist_rand_vect)
    
    # For entity_1 and entity_2
    # pretrained_embedding_matrix.append(rand_vect)
    # pretrained_embedding_matrix.append(rand_vect)
    
    unk_word_count = 0
    word_count = len(word_to_ix)
         
    for word, word_idx in word_to_ix.items():
        if word in [UNKNOWN_WORD, PADDING_WORD]:
            continue
        try:
            w2v_model[word]
        except KeyError:
            unk_word_count += 1
            # print(f"found UNK: {word}")
            pretrained_embedding_matrix.append(rand_vect)
        else:
            pretrained_embedding_matrix.append(np.array(w2v_model[word]))
    print(f"\nTotal UNK words: {unk_word_count}/{word_count}")
        
    for dist, dist_idx in distance_to_ix.items():
        if dist in [UNKNOWN_WORD, PADDING_WORD]:
            continue
        
        distance_pretrain_embedding_matrix.append(np.full(dist_dim, np.tanh(dist/max_distance)))
    
    # For unknown word
    pretrained_embedding_matrix.append(rand_vect)
    distance_pretrain_embedding_matrix.append(dist_rand_vect)
    
    if len(word_to_ix) != len(pretrained_embedding_matrix):
        raise ValueError(f'Word_to_ix (size: {len(word_to_ix)}) not equals Pretrained_embedding_matrix (size: {len(pretrained_embedding_matrix)})')
    
    return np.array(pretrained_embedding_matrix), np.array(distance_pretrain_embedding_matrix)
                                  