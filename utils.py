from imports import *
from constants import *

def preprocess_include_cross_sentence(file_path, synonym_dict):
    
    dataloaders = {
        'train': [],
        'dev': [],
        'test': []
    }

    for phase in ['train', 'dev', 'test']:
        print(f"** {phase[0].upper()}{phase[1:]}")
        file = minidom.parse(f"{file_path}BioNLP-ST-2016_BB-event_{phase}.xml")
        docs = file.getElementsByTagName("document")
        tmp_data = []
        for doc in docs:
            document_idx = doc.getAttribute("origId")
            sentences = doc.getElementsByTagName("sentence")
            for sentence in sentences:

                current_sentence_idx = sentence.getAttribute("id")

                # Collect corresponding sentence_idx in the relations
                sentence_idx_set = set()
                interactions = sentence.getElementsByTagName("interaction")
                for interaction in interactions:
                    entity_1 = interaction.getAttribute("e1")
                    entity_2 = interaction.getAttribute("e2")
                    first_sentence_idx = re.match(r'(TEES.d\d+.s\d+).e\d+', entity_1).group(1)
                    second_sentence_idx = re.match(r'(TEES.d\d+.s\d+).e\d+', entity_2).group(1)
                    sentence_idx_set.add(first_sentence_idx)
                    sentence_idx_set.add(second_sentence_idx)

                if phase == 'test':
                    sentence_format = re.match(r'(TEES.d\d+.s)\d+', current_sentence_idx).group(1)
                    sentence_count = int(re.match(r'TEES.d\d+.s(\d+)', current_sentence_idx).group(1))
                    if (sentence_count-1)>=0:
                        sentence_idx_set.add(f"{sentence_format}{sentence_count-1}")
                    sentence_idx_set.add(f"{sentence_format}{sentence_count}")
                    sentence_idx_set.add(f"{sentence_format}{sentence_count+1}")

                # Extract features for the corresponding sentence
                sentence_elements = {}
                for sentence_idx in list(sentence_idx_set):
                    for tmp_sentence in sentences:
                        tmp_sentence_idx = tmp_sentence.getAttribute("id")
                        if sentence_idx == tmp_sentence_idx:
                            entities = tmp_sentence.getElementsByTagName("entity")
                            analyses = tmp_sentence.getElementsByTagName("analyses")[0]
                            tokens = analyses.getElementsByTagName("tokenization")[0].getElementsByTagName("token")
                            dependencies = tmp_sentence.getElementsByTagName("parse")[0].getElementsByTagName("dependency")

                            sentence_elements[sentence_idx] = {
                                'tokens': tokens,
                                'entities': entities,
                                'dependencies': dependencies,
                            }

                            break

                # Extract entity pairs
                entity_pairs, entity_to_charoffset, entity_idx_to_type, entity_idx_to_origId, entity_idx_to_text = _generate_entity_pairs(sentence_elements)

                # Extract label pairs
                label_pairs = []
                tmp_label_pairs = []
                for interaction in interactions:
                    tmp_label_pairs.append((interaction.getAttribute("e1"), interaction.getAttribute("e2")))

                for (entity_pair, sentence_idx_pair) in entity_pairs:
                    if entity_pair in tmp_label_pairs:
                        label_pairs.append(1)
                    else:
                        label_pairs.append(0)

                # Iterates entity_pairs
                full_inputs = []
                shortest_inputs = []
                entity_token_pairs = []

                for idx, ((first_entity, second_entity), (first_sentence_idx, second_sentence_idx)) in enumerate(entity_pairs):

                    if first_sentence_idx == second_sentence_idx:
                        shortest_input, full_input, entity_token_pair = _extract_in_sentence_feature(sentence_elements, first_entity, second_entity, current_sentence_idx, entity_to_charoffset, entity_idx_to_text, synonym_dict)
                        if full_input and shortest_input:
                            full_inputs.append(full_input)
                            shortest_inputs.append(shortest_input)
                            entity_token_pairs.append(entity_token_pair)
                    else:
                        if first_sentence_idx > second_sentence_idx: # second_entity is mentioned before first_entity
                            shortest_input, full_input, entity_token_pair = _extract_cross_sentence_feature(sentence_elements, second_entity, first_entity, second_sentence_idx, first_sentence_idx, entity_to_charoffset, entity_idx_to_text, synonym_dict)
                        else: # first_entity is mentioned before second_entity
                            shortest_input, full_input, entity_token_pair = _extract_cross_sentence_feature(sentence_elements, first_entity, second_entity, first_sentence_idx, second_sentence_idx, entity_to_charoffset, entity_idx_to_text, synonym_dict)

                        if full_input and shortest_input:
                            full_inputs.append(full_input)
                            shortest_inputs.append(shortest_input)
                            entity_token_pairs.append(entity_token_pair)

                entity_pairs = [entity_pair[0] for entity_pair in entity_pairs]           
                for idx, shortest_input in enumerate(shortest_inputs):
                    dataloaders[phase].append({
                        'document_id': document_idx,
                        'sentence_id': sentence_idx,
                        'shortest_inputs': shortest_input,
                        'full_inputs': full_inputs[idx],
                        'entity_pair': entity_pairs[idx], # ordered by bacteria and location
                        'entity_token_pair': entity_token_pairs[idx], # ordered by prev_sent and current_sent
                        'label': label_pairs[idx],
                        'entity_idx_to_type': entity_idx_to_type,
                        'entity_idx_to_origId': entity_idx_to_origId,
                    }) 
        print()
        
    return dataloaders

def preprocess(file_path, synonym_dict):
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
                                        # "entity_1"
                                        try:
                                            # token['text'].lower()
                                            # entity_idx_to_text[start_entity].lower()
                                            synonym = synonym_dict[token['text'].lower()]
                                        except KeyError:
                                            tkn = token['text'].lower()
                                        else:
                                            # print("[SYNONYM]: Replace `{}` with `{}`".format(token['text'].lower(), synonym))
                                            tkn = synonym
                                        shortest_token_list.append(token['text'].lower())
                                        # shortest_token_list.append('entity_1')

                                            
                                    elif token['id'] == end_token:
                                        # "entity_2"
                                        try:
                                            synonym = synonym_dict[token['text'].lower()]
                                        except KeyError:
                                            tkn = token['text'].lower()
                                        else:
                                            # print("[SYNONYM]: replace `{}` with `{}`".format(token['text'].lower(), synonym))
                                            tkn = synonym
                                        shortest_token_list.append(token['text'].lower())
                                        # shortest_token_list.append('entity_2')

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
                            # "entity_1"
                            try:
                                synonym = synonym_dict[token['text'].lower()]
                            except KeyError:
                                tkn = token['text'].lower()
                            else:
                                # print("[SYNONYM]: Replace `{}` with `{}`".format(token['text'].lower(), synonym))
                                tkn = synonym
                            full_token_list.append(token['text'].lower())
                            # full_token_list.append('entity_1')
                        elif token['id'] == end_token:
                            # "entity_2"
                            try:
                                synonym = synonym_dict[token['text'].lower()]
                            except KeyError:
                                tkn = token['text'].lower()
                            else:
                                # print("[SYNONYM]: Replace `{}` with `{}`".format(token['text'].lower(), synonym))
                                tkn = synonym
                            full_token_list.append(token['text'].lower())
                            # full_token_list.append('entity_2')
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


def build_vocab(dataloaders, w2v_model, top_k=100000, types='shortest'):
    word_count = {}
    for word, vector in w2v_model.vocab.items():
        word_count[word] = w2v_model.vocab[word].count
    vocabs = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
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
    
    word_to_ix['entity_1'] = len(word_to_ix)
    word_to_ix['entity_2'] = len(word_to_ix)
    
    for vocab, count in vocabs:
        word_to_ix[vocab] = len(word_to_ix)
    
    for input_dict in dataloaders['train']+dataloaders['dev']:
        inputs = input_dict[f'{types}_inputs']
        for token in inputs[f'{types}_token']:
            if token not in word_to_ix:
                # print("OOV:", token)
                in_vocab_count['OOV'] += 1
            else:
                in_vocab_count['VOCAB'] += 1
                
            for char in token:
                if char not in char_to_ix:
                    char_to_ix[char] = len(char_to_ix)
        for pos in inputs[f'{types}_pos']:
            if pos not in pos_to_ix:
                pos_to_ix[pos] = len(pos_to_ix)
        for dep in inputs[f'{types}_dep']:
            if dep not in dependency_to_ix:
                dependency_to_ix[dep] = len(dependency_to_ix)
        for dist in inputs[f'{types}_dist1']+inputs[f'{types}_dist2']:
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
    pretrained_embedding_matrix.append(rand_vect)
    pretrained_embedding_matrix.append(rand_vect)
    
    for word, word_idx in word_to_ix.items():
        if word in [UNKNOWN_WORD, PADDING_WORD, 'entity_1', 'entity_2']:
            continue
        pretrained_embedding_matrix.append(np.array(w2v_model[word]))
        
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
                                  

def _find_root(G,child):
    parent = list(G.predecessors(child))
    if len(parent) == 0:
        return child
    else:  
        return _find_root(G, parent[0])
                                  
def _generate_entity_pairs(sentence_elements):
    
    entity_type_dict = {
        'Habitat': set(),
        'Bacteria': set(),
        'Geographical': set()
    }
    
    label_pairs = []
    entity_pairs = []
    entity_to_charoffset = {}
    entity_idx_to_type = {}
    entity_idx_to_origId = {}
    entity_idx_to_text = {}

    for sentence_idx, sentence_element in sentence_elements.items():
        for entity in sentence_element['entities']:
            entity_type = entity.getAttribute("type")
            entity_origId = entity.getAttribute("origId")
            entity_idx = entity.getAttribute("id")
            
            if entity_type in entity_type_dict:
                entity_type_dict[entity_type].add(entity_idx)
                entity_idx_to_type[entity_idx] = entity_type
                entity_idx_to_origId[entity_idx] = entity_origId
                entity_to_charoffset[entity_idx] = entity.getAttribute("headOffset")
                entity_idx_to_text[entity_idx] = entity.getAttribute("text")

    for bacteria in list(entity_type_dict['Bacteria']):
        for location in list(entity_type_dict['Habitat'])+list(entity_type_dict['Geographical']):
            # Exclude cross-sentence relations
            try:
                first_sentence_idx = re.match(r'(TEES.d\d+.s\d+).e\d+', bacteria).group(1)
                second_sentence_idx = re.match(r'(TEES.d\d+.s\d+).e\d+', location).group(1)
            except AttributeError:
                continue
            else:
                entity_pairs.append([(bacteria, location), (first_sentence_idx, second_sentence_idx)])
    
    return entity_pairs, entity_to_charoffset, entity_idx_to_type, entity_idx_to_origId, entity_idx_to_text
                                  
                                  
def _extract_in_sentence_feature(sentence_elements, first_entity, second_entity, current_sentence_idx, entity_to_charoffset, entity_idx_to_text, synonym_dict):
    entity_token_pair = ["", ""]
    
    token_list = []
    for token in sentence_elements[current_sentence_idx]['tokens']:
        tmp_token_dict = {}
        tmp_token_dict['id'] = token.getAttribute("id")
        tmp_token_dict['text'] = token.getAttribute("text")
        tmp_token_dict['charOffset'] = token.getAttribute("charOffset")
        tmp_token_dict['POS'] = token.getAttribute("POS")

        token_list.append(tmp_token_dict)
        
    dep_list = []
    edges = []
    for dependency in sentence_elements[current_sentence_idx]['dependencies']:
        edges.append((dependency.getAttribute("t1"), dependency.getAttribute("t2")))

        dep_list.append({
            'id': dependency.getAttribute("id"),
            'path': (dependency.getAttribute("t1"), dependency.getAttribute("t2")),
            'type': dependency.getAttribute("type")
        })
    
    start_token = entity_idx_to_text[first_entity]
    end_token = entity_idx_to_text[second_entity]
    for token in token_list:
        if token['charOffset'] == entity_to_charoffset[first_entity]:
            start_token = token['id']
        if token['charOffset'] == entity_to_charoffset[second_entity]:
            end_token = token['id']
            
    graph = nx.Graph(edges)
    
    try:
        shortest_path = nx.shortest_path(graph, source=start_token, target=end_token)
    except nx.NetworkXNoPath:
        print(f"-- NetworkXNoPath: between {start_token} and {end_token} --")
        shortest_path = None
    except nx.NodeNotFound:
        print(f"-- NodeNotFound: {start_token} and {end_token} --")
        shortest_path = None
    else:
        if len(shortest_path) == 1:
            print(f"-- ShortestPathOnlyOneItem (entity pair is the same words): between {start_token} and {end_token} --")
            shortest_path = None
    
    # Find entity_1 and entity_2 idx to calculate distance
    entity_1_idx = 0
    entity_2_idx = 0
    for idx, token in enumerate(token_list): 
        if token['id'] == start_token:
            entity_1_idx = idx
        if token['id'] == end_token: 
            entity_2_idx = idx
    
    # Define shortest and full texts
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
                        # "entity_1"
                        try:
                            # token['text'].lower()
                            synonym = synonym_dict[token['text'].lower()]
                        except KeyError:
                            tkn = token['text'].lower()
                        else:
                            # print("[SYNONYM]: Replace `{}` with `{}`".format(token['text'].lower(), synonym))
                            tkn = synonym
                        shortest_token_list.append(tkn)
                        # shortest_token_list.append('entity_1')
                        entity_token_pair[0] = tkn


                    elif token['id'] == end_token:
                        # "entity_2"
                        try:
                            synonym = synonym_dict[token['text'].lower()]
                        except KeyError:
                            tkn = token['text'].lower()
                        else:
                            # print("[SYNONYM]: replace `{}` with `{}`".format(token['text'].lower(), synonym))
                            tkn = synonym
                        shortest_token_list.append(tkn)
                        # shortest_token_list.append('entity_2')
                        entity_token_pair[1] = tkn

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
        print(f"### Shortest not found! between {start_token} and {end_token} ###")
        return None, None, None
        
    # Full texts
    for idx, token in enumerate(token_list):
        if token['id'] == start_token: # "entity_1"
            try:
                synonym = synonym_dict[token['text'].lower()]
            except KeyError:
                tkn = token['text'].lower()
            else:
                # print("[SYNONYM]: Replace `{}` with `{}`".format(token['text'].lower(), synonym))
                tkn = synonym
            full_token_list.append(tkn)
            # full_token_list.append('entity_1')
        elif token['id'] == end_token: # "entity_2"     
            try:
                synonym = synonym_dict[token['text'].lower()]
            except KeyError:
                tkn = token['text'].lower()
            else:
                # print("[SYNONYM]: Replace `{}` with `{}`".format(token['text'].lower(), synonym))
                tkn = synonym
            full_token_list.append(tkn)
            # full_token_list.append('entity_2')
        else:
            full_token_list.append(token['text'].lower())
        full_pos_list.append(token['POS'])
        full_dist1_list.append(entity_1_idx-idx)
        full_dist2_list.append(entity_2_idx-idx)
    for idx, dep in enumerate(dep_list):
            full_dep_list.append(dep['type'])    
    
    shortest_input = {
        'shortest_token': shortest_token_list,
        'shortest_pos': shortest_pos_list,
        'shortest_dist1': shortest_dist1_list,
        'shortest_dist2': shortest_dist2_list,
        'shortest_dep': shortest_dep_list
    }
    
    full_input = {
        'full_token': full_token_list,
        'full_pos': full_pos_list,
        'full_dist1': full_dist1_list,
        'full_dist2': full_dist2_list,
        'full_dep': full_dep_list
    }
    
    return shortest_input, full_input, entity_token_pair
                                  
                                  
def _extract_cross_sentence_feature(sentence_elements, first_entity, second_entity, first_sentence_idx, second_sentence_idx, entity_to_charoffset, entity_idx_to_text, synonym_dict):
    
    entity_token_pair = ["", ""]

    prev_tokens = sentence_elements[first_sentence_idx]['tokens']
    prev_dependencies = sentence_elements[first_sentence_idx]['dependencies']
    next_tokens = sentence_elements[second_sentence_idx]['tokens']
    next_dependencies = sentence_elements[second_sentence_idx]['dependencies']
        
    # Define shortest and full texts
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
    
    # Extract feature for the first sentence
    token_list = []
    for token in prev_tokens:
        tmp_token_dict = {}
        tmp_token_dict['id'] = token.getAttribute("id")
        tmp_token_dict['text'] = token.getAttribute("text")
        tmp_token_dict['charOffset'] = token.getAttribute("charOffset")
        tmp_token_dict['POS'] = token.getAttribute("POS")

        token_list.append(tmp_token_dict)
        
    dep_list = []
    edges = []
    for dependency in prev_dependencies:
        edges.append((dependency.getAttribute("t1"), dependency.getAttribute("t2")))

        dep_list.append({
            'id': dependency.getAttribute("id"),
            'path': (dependency.getAttribute("t1"), dependency.getAttribute("t2")),
            'type': dependency.getAttribute("type")
        })
    
    start_token = ""
    end_token = ""
    for token in token_list:
        if token['charOffset'] == entity_to_charoffset[first_entity]:
            start_token = token['id']
        if token['text'] == '.':
            end_token = token['id']
     
    di_graph = nx.DiGraph(edges)
    try:
        end_token = _find_root(di_graph, start_token) # root word (usually main verb)
    except RecursionError:
        print(f"-- RecursionError: can not find root word for {start_token} use `.` instead --")
    
    graph = nx.Graph(edges)
    try:
        shortest_path = nx.shortest_path(graph, source=start_token, target=end_token)
    except nx.NetworkXNoPath:
        print(f"-- NetworkXNoPath: between {start_token} and {end_token} --")
        shortest_path = None
    except nx.NodeNotFound:
        print(f"-- NodeNotFound: {start_token} and {end_token} --")
        shortest_path = None
    else:
        if len(shortest_path) == 1:
            print(f"-- ShortestPathOnlyOneItem (entity pair is the same words): between {start_token} and {end_token} --")
            shortest_path = None
    
    # Find entity_1 and entity_2 idx to calculate distance
    entity_1_idx = 0
    entity_2_idx = 0
    for idx, token in enumerate(token_list): 
        if token['id'] == start_token:
            entity_1_idx = idx
        if token['id'] == end_token: 
            entity_2_idx = idx

    if shortest_path:
        # print("### Shortest! ###")
        for token_idx in shortest_path:
            for idx, token in enumerate(token_list):
                if token['id'] == token_idx:
                    if token['id'] == start_token:
                        # "entity_1"
                        try:
                            # token['text'].lower()
                            synonym = synonym_dict[token['text'].lower()]
                        except KeyError:
                            tkn = token['text'].lower()
                        else:
                            # print("[SYNONYM]: Replace `{}` with `{}`".format(token['text'].lower(), synonym))
                            tkn = synonym
                        shortest_token_list.append(tkn)
                        # shortest_token_list.append('entity_1')
                        entity_token_pair[0] = tkn

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
        print(f"-- Shortest not found! between {start_token} and {end_token} --")
        return None, None, None
        
    # Full texts
    for idx, token in enumerate(token_list):
        if token['id'] == start_token: # "entity_1"
            try:
                synonym = synonym_dict[token['text'].lower()]
            except KeyError:
                tkn = token['text'].lower()
            else:
                # print("[SYNONYM]: Replace `{}` with `{}`".format(token['text'].lower(), synonym))
                tkn = synonym
            full_token_list.append(tkn)
            # full_token_list.append('entity_1')

        else:
            full_token_list.append(token['text'].lower())
        full_pos_list.append(token['POS'])
        full_dist1_list.append(entity_1_idx-idx)
        full_dist2_list.append(entity_2_idx-idx)
    for idx, dep in enumerate(dep_list):
            full_dep_list.append(dep['type'])    
    
    # shortest_token_list.append('[SEP]')
    # shortest_pos_list.append('[SEP]')
    # shortest_dist1_list.append('[SEP]')
    # shortest_dist2_list.append('[SEP]')
    # shortest_dep_list.append('[SEP]')

    # full_token_list.append('[SEP]')
    # full_pos_list.append('[SEP]')
    # full_dist1_list.append('[SEP]')
    # full_dist2_list.append('[SEP]')
    # full_dep_list.append('[SEP]')

    # Extract feature for the second sentence
    token_list = []
    for token in next_tokens:
        tmp_token_dict = {}
        tmp_token_dict['id'] = token.getAttribute("id")
        tmp_token_dict['text'] = token.getAttribute("text")
        tmp_token_dict['charOffset'] = token.getAttribute("charOffset")
        tmp_token_dict['POS'] = token.getAttribute("POS")

        token_list.append(tmp_token_dict)
        
    dep_list = []
    edges = []
    for dependency in next_dependencies:
        edges.append((dependency.getAttribute("t1"), dependency.getAttribute("t2")))

        dep_list.append({
            'id': dependency.getAttribute("id"),
            'path': (dependency.getAttribute("t1"), dependency.getAttribute("t2")),
            'type': dependency.getAttribute("type")
        })
    
    start_token = ""
    end_token = ""
    for token in token_list:
        if token['charOffset'] == entity_to_charoffset[second_entity]:
            start_token = token['id']
        if token['text'] == '.':
            end_token = token['id']
     
    di_graph = nx.DiGraph(edges)
    try:
        end_token = _find_root(di_graph, start_token) # root word (usually main verb)
    except RecursionError:
        print(f"-- RecursionError: can not find root word for {start_token} use `.` instead --")
    
    graph = nx.Graph(edges)
    try:
        shortest_path = nx.shortest_path(graph, source=start_token, target=end_token)
    except nx.NetworkXNoPath:
        print(f"-- NetworkXNoPath: between {start_token} and {end_token}--")
        shortest_path = None
    except nx.NodeNotFound:
        print(f"-- NodeNotFound: {start_token} and {end_token} --")
        shortest_path = None
    else:
        if len(shortest_path) == 1:
            print(f"-- ShortestPathOnlyOneItem (entity pair is the same words): between {start_token} and {end_token}--")
            shortest_path = None

    # Find entity_1 and entity_2 idx to calculate distance
    entity_1_idx = 0
    entity_2_idx = 0
    for idx, token in enumerate(token_list): 
        if token['id'] == start_token:
            entity_1_idx = idx
        if token['id'] == end_token: 
            entity_2_idx = idx

    if shortest_path:
        # print("### Shortest! ###")
        for token_idx in shortest_path:
            for idx, token in enumerate(token_list):
                if token['id'] == token_idx:
                    if token['id'] == start_token:
                        # "entity_1"
                        try:
                            # token['text'].lower()
                            synonym = synonym_dict[token['text'].lower()]
                        except KeyError:
                            tkn = token['text'].lower()
                        else:
                            # print("[SYNONYM]: Replace `{}` with `{}`".format(token['text'].lower(), synonym))
                            tkn = synonym
                        shortest_token_list.append(tkn)
                        # shortest_token_list.append('entity_2')
                        entity_token_pair[1] = tkn

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
        print(f"-- Shortest not found! between {start_token} and {end_token} --")
        return None, None, None
        
    # Full texts
    for idx, token in enumerate(token_list):
        if token['id'] == start_token: # "entity_2"
            try:
                synonym = synonym_dict[token['text'].lower()]
            except KeyError:
                tkn = token['text'].lower()
            else:
                # print("[SYNONYM]: Replace `{}` with `{}`".format(token['text'].lower(), synonym))
                tkn = synonym
            full_token_list.append(tkn)
            # full_token_list.append('entity_2')

        else:
            full_token_list.append(token['text'].lower())
        full_pos_list.append(token['POS'])
        full_dist1_list.append(entity_1_idx-idx)
        full_dist2_list.append(entity_2_idx-idx)
    for idx, dep in enumerate(dep_list):
            full_dep_list.append(dep['type'])  
    
    # Return
    shortest_input = {
        'shortest_token': shortest_token_list,
        'shortest_pos': shortest_pos_list,
        'shortest_dist1': shortest_dist1_list,
        'shortest_dist2': shortest_dist2_list,
        'shortest_dep': shortest_dep_list
    }
    
    full_input = {
        'full_token': full_token_list,
        'full_pos': full_pos_list,
        'full_dist1': full_dist1_list,
        'full_dist2': full_dist2_list,
        'full_dep': full_dep_list
    }
    
    return shortest_input, full_input, entity_token_pair