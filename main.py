from imports import *
from utils import *
from constants import *
from models import *
from trains import train_model
from predicts import predict_model

import pickle
import shutil
from tqdm import tqdm
from time import sleep
import json_lines
import os

if not os.path.exists(OUTPUT_DIR_PATH):
    print(f"Create directory: {OUTPUT_DIR_PATH}")
    os.makedirs(OUTPUT_DIR_PATH)

torch.cuda.set_device(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataloaders = {}

with open(os.path.join(DATA_DIR_PATH, "preprocessed_BB/train.json"), "r") as read_file:
    dataloaders['train'] = json.load(read_file)

with open(os.path.join(DATA_DIR_PATH, "preprocessed_BB/dev.json"), "r") as read_file:
    dataloaders['dev'] = json.load(read_file)

with open(os.path.join(DATA_DIR_PATH, "preprocessed_BB/test.json"), "r") as read_file:
    dataloaders['test'] = json.load(read_file)

# Load w2v model
w2v_model = word2vec.KeyedVectors.load_word2vec_format(W2V_MODEL_PATH, binary=True)

# # Global max relative distance
max_distance = float(np.max([np.max(np.abs(input_dict['full_inputs']['full_dist1'] +
                                           input_dict['full_inputs']['full_dist2'])) for input_dict in dataloaders['train']+dataloaders['dev']]))

word_to_ix, pos_to_ix, distance_to_ix, dependency_to_ix, char_to_ix, in_vocab_count = build_vocab(
    dataloaders, w2v_model)

glob_shortest_max_sentence_length = np.max([np.max(
    [len(input_dict['shortest_inputs']['shortest_token']),
     len(input_dict['shortest_inputs']['shortest_pos']),
     len(input_dict['shortest_inputs']['shortest_dep'])]) for input_dict in dataloaders['train']+dataloaders['dev']])

glob_max_sentence_length = np.max([np.max(
    [len(input_dict['full_inputs']['full_token']),
     len(input_dict['full_inputs']['full_pos']),
     len(input_dict['full_inputs']['full_dep'])]) for input_dict in dataloaders['train']+dataloaders['dev']])

pretrained_embedding_matrix, distance_pretrain_embedding_matrix = build_pretrain_embedding_matrix(
    w2v_model, word_to_ix, distance_to_ix, max_distance)

batch_size = 4

f1_writer = []
model_idx = 0

elmo_model = Elmo(ELMO_OPTIONS_PATH, ELMO_MODEL_PATH, 1, dropout=0)

# BERT features

finetune_berts = []
with open(BERT_FEATURES_PATH, 'rb') as f:
    for item in json_lines.reader(f):
        finetune_berts.append(item)

with open(BERT_TKN_PATH, 'r') as f:
    full_bert_tkns = f.read()

for idx, full_bert_tkn in enumerate(full_bert_tkns.split('\n')):
    for dataloader in dataloaders['train']+dataloaders['dev']+dataloaders['test']:
        if " ".join(dataloader['full_inputs']['full_token']) == full_bert_tkn:
            dataloader['bert_features'] = np.sum([np.array(
                layer['values']) for layer in finetune_berts[0]['features'][0]['layers']], axis=0)

# In-side For loop
for seed in tqdm(SEED_LIST[20:22], desc='seed loop'):

    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = Frankenstein(len(word_to_ix), len(pos_to_ix), len(distance_to_ix), len(dependency_to_ix),
                         glob_max_sentence_length, pretrained_embedding_matrix,
                         distance_pretrain_embedding_matrix,
                         batch_size, drop=0.5, wdrop=0.3, edrop=0.3, idrop=0.3,
                         hidden_dim=64, window_sizes=[3, 5, 7], h=1, multihead_sizes=3)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    train_out = train_model(model, elmo_model, dataloaders['train'], dataloaders['dev'], word_to_ix,
                            pos_to_ix,
                            distance_to_ix, dependency_to_ix, criterion, optimizer_ft,
                            lr_scheduler=None,
                            num_epochs=20, early_stopped_patience=1, batch_size=batch_size,
                            verbose=True)

    (model, train_f1, val_f1, history) = train_out

    y_true, y_pred, predictions = predict_model(
        model, elmo_model, dataloaders['test'], word_to_ix, pos_to_ix, distance_to_ix, dependency_to_ix, char_to_ix, batch_size, optimizer_ft)

    test_data = dataloaders['test']
    file = minidom.parse(os.path.join(
        DATA_DIR_PATH, "BioNLP-ST-2016_BB-event_test.xml"))
    docs = file.getElementsByTagName("document")
    all_test_files = []
    for doc in docs:
        all_test_files.append(doc.getAttribute("origId"))

    model_dir_name = f"model_{model_idx}"
    sleep(0.1)

    output_dir_path = os.path.join(OUTPUT_DIR_PATH, model_dir_name)
    if not os.path.exists(output_dir_path):
        print(f"Create directory: {output_dir_path}")
        os.makedirs(output_dir_path)

    relation_idx_dict = {}
    pred_test_files = set()
    for idx, input_dict in enumerate(test_data):
        inputs = input_dict['shortest_inputs']
        entity_tag = input_dict['entity_pair']
        label = input_dict['label']
        entity_idx_to_type = input_dict['entity_idx_to_type']

        if y_pred[idx] == 1:
            document_idx = input_dict['document_id']
            pred_test_files.add(document_idx)
            entity_idx_to_origId = input_dict['entity_idx_to_origId']
            first_match = re.match(
                r'(BB-event-\d+).(T\d+)', entity_idx_to_origId[entity_tag[0]])
            second_match = re.match(
                r'(BB-event-\d+).(T\d+)', entity_idx_to_origId[entity_tag[1]])

            first_entity = first_match.group(2).upper()
            second_entity = second_match.group(2).upper()
            first_doc = first_match.group(1)
            second_doc = second_match.group(1)
            if first_doc == second_doc:
                try:
                    relation_idx_dict[document_idx] += 1
                except KeyError:
                    relation_idx_dict[document_idx] = 1

                f = open(os.path.join(output_dir_path,
                                      f"{document_idx}.a2"), "a+")
                f.write("R{}\tLives_In Bacteria:{} Location:{}\n".format(
                    relation_idx_dict[document_idx], first_entity, second_entity))
                f.close()

    pred_test_files = list(pred_test_files)
    for test_file in all_test_files:
        if not test_file in pred_test_files:
            f = open(os.path.join(output_dir_path, f"{test_file}.a2"), "a+")
            f.write("")
            f.close()

    sleep(0.1)

    shutil.make_archive(output_dir_path, 'zip', output_dir_path)

    model_idx += 1

    f1_writer.append({
        'model_name': model_dir_name,
        'seed': seed
    })

    with open(os.path.join(output_dir_path, "log.json"), 'w') as outfile:
        json.dump(f1_writer, outfile)
