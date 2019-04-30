from imports import *
from constants import *
from utils import *


def predict_model(model, elmo_model, test_data, word_to_ix, pos_to_ix,
                  distance_to_ix, batch_size, optimizer):

    model.eval()

    y_true = []
    y_pred = []
    predictions = []
    
    test_dataloaders = []

    for i in range(0, len(test_data), batch_size):
        test_batch_data = np.array(test_data)[i:i+batch_size]

        token_batch = []
        pos_batch = []
        dist1_batch = []
        dist2_batch = []
        lengths_batch = []

        shortest_token_batch = []
        shortest_pos_batch = []
        shortest_dist1_batch = []
        shortest_dist2_batch = []
        shortest_position_batch = []
        shortest_lengths_batch = []

        label_batch = []

        entity_batch = []
        
        test_dataloader_batch = []

        max_sentence_length = np.max([np.max(
                [len(input_dict['full_inputs']['full_token']), 
                 len(input_dict['full_inputs']['full_pos']), 
                 len(input_dict['full_inputs']['full_dist1']), 
                 len(input_dict['full_inputs']['full_dist2'])]) for input_dict in test_batch_data])
        shortest_max_sentence_length = np.max([np.max([len(input_dict['shortest_inputs']['shortest_token']), 
                                                       len(input_dict['shortest_inputs']['shortest_pos']), 
                                                       len(input_dict['shortest_inputs']['shortest_dist1']), 
                                                       len(input_dict['shortest_inputs']['shortest_dist2'])]) for input_dict in test_batch_data])

        ELMO_sentences = []
        ELMO_shortest = []
        ELMO_entities = []
        BERT_features = []

        for input_dict in test_batch_data:
            
            test_dataloader_batch.append(input_dict)
            
            inputs = input_dict['full_inputs']
            shortset_inputs = input_dict['shortest_inputs']
            entity_tag = input_dict['entity_pair']
            label = input_dict['label']
            entity_idx_to_type = input_dict['entity_idx_to_type']

            token, token_idx, lengths = prepare_sequence(inputs['full_token'],
                                                         word_to_ix, max_length=max_sentence_length,
                                                         padding=True, get_lengths=True)
            pos, pos_idx = prepare_sequence(inputs['full_pos'],
                                            pos_to_ix, max_length=max_sentence_length, prepare_word=False, padding=True)
            dist1, dist1_idx = prepare_sequence(inputs['full_dist1'],
                                                distance_to_ix, max_length=max_sentence_length,
                                                prepare_word=False, padding=True)
            dist2, dist2_idx = prepare_sequence(inputs['full_dist2'],
                                                distance_to_ix, max_length=max_sentence_length,
                                                prepare_word=False, padding=True)

            ELMO_sentences.append(token)

            token_batch.append(token_idx)
            pos_batch.append(pos_idx)
            dist1_batch.append(dist1_idx)
            dist2_batch.append(dist2_idx)
            label_batch.append(torch.tensor([label], dtype=torch.long))
            lengths_batch.append(lengths)

            token, token_idx, shortest_lengths = prepare_sequence(shortset_inputs['shortest_token'],
                                                                  word_to_ix, max_length=shortest_max_sentence_length, padding=True, get_lengths=True)
            pos, pos_idx = prepare_sequence(shortset_inputs['shortest_pos'],
                                            pos_to_ix, max_length=shortest_max_sentence_length,
                                            prepare_word=False, padding=True)
            dist1, dist1_idx = prepare_sequence(shortset_inputs['shortest_dist1'],
                                                distance_to_ix, max_length=shortest_max_sentence_length,
                                                prepare_word=False, padding=True)
            dist2, dist2_idx = prepare_sequence(shortset_inputs['shortest_dist2'],
                                                distance_to_ix, max_length=shortest_max_sentence_length,
                                                prepare_word=False, padding=True)

            ELMO_shortest.append(token)

            position_idx = []
            for i in range(shortest_lengths):
                position_idx.append(i)
            for _ in range(shortest_max_sentence_length-shortest_lengths):
                position_idx.append(0)

            shortest_position_batch.append(
                torch.tensor(position_idx, dtype=torch.long))

            shortest_token_batch.append(token_idx)
            shortest_pos_batch.append(pos_idx)
            shortest_dist1_batch.append(dist1_idx)
            shortest_dist2_batch.append(dist2_idx)
            shortest_lengths_batch.append(shortest_lengths)

            # Entity and Is_a relation for each iter
            _, entities = prepare_sequence(["entity_1", "entity_2"], word_to_ix, max_length=2)

            entity_batch.append(entities)

            ELMO_entities.append(["entity_1", "entity_2"])

            BERT_features.append(torch.tensor(input_dict['bert_features'], dtype=torch.float))

        # Sort lengths in decending order
        inds = np.argsort(-torch.stack(lengths_batch).numpy())
        label_batch = torch.stack(label_batch)[inds].cuda()
        lengths_batch = torch.stack(lengths_batch)[inds].cuda()
        
        test_dataloader_batch = np.array(test_dataloader_batch)[inds]

        batch_in = {
            'token': torch.stack(token_batch)[inds].cuda(),
            'pos': torch.stack(pos_batch)[inds].cuda(),
            'dist1': torch.stack(dist1_batch)[inds].cuda(),
            'dist2': torch.stack(dist2_batch)[inds].cuda(),
            'bert_features': torch.stack(BERT_features)[inds].cuda()
        }

        shortest_batch_in = {
            'token': torch.stack(shortest_token_batch)[inds].cuda(),
            'pos': torch.stack(shortest_pos_batch)[inds].cuda(),
            'dist1': torch.stack(shortest_dist1_batch)[inds].cuda(),
            'dist2': torch.stack(shortest_dist2_batch)[inds].cuda(),
            'position': torch.stack(shortest_position_batch)[inds].cuda()
        }

        character_ids = batch_to_ids(ELMO_sentences)
        ELMO_embeddings = elmo_model(character_ids)['elmo_representations'][0].cuda()

        character_ids = batch_to_ids(ELMO_shortest)
        ELMO_shortest_embeddings = elmo_model(character_ids)['elmo_representations'][0].cuda()

        character_ids = batch_to_ids(ELMO_entities)
        ELMO_entity_embeddings = elmo_model(character_ids)['elmo_representations'][0].cuda()

        entities = torch.stack(entity_batch)[inds]

        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = model(batch_in, shortest_batch_in, entities, lengths_batch, 
                        ELMO_embeddings, 
                        ELMO_shortest_embeddings, 
                        ELMO_entity_embeddings)

        if len(outputs.size()) == 1:  # batch_size is 1
            outputs = outputs.unsqueeze(0)

        predictions.append(outputs)
        preds = torch.max(outputs, dim=1)[1]

        y_true += [x.item() for x in label_batch]
        y_pred += [x.item() for x in preds]
        
        test_dataloaders += [x for x in test_dataloader_batch]

    return np.array(y_true), np.array(y_pred), test_dataloaders
