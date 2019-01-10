from imports import *
from constants import *
from utils import *


def predict_model(model, test_data, word_to_ix, pos_to_ix,
                  distance_to_ix, dependency_to_ix,
                  char_to_ix, batch_size, optimizer):

    model.eval()

    y_true = []
    y_pred = []
    predictions = []

    self_attn_scores = []
    ent_attn_scores = []
    multihead_attn_scores = []

    for i in range(0, len(test_data), batch_size):
        test_batch_data = np.array(test_data)[i:i+batch_size]

        shortest_token_batch = []
        shortest_pos_batch = []
        shortest_dist1_batch = []
        shortest_dist2_batch = []
        shortest_lengths_batch = []

        label_batch = []

        shortest_max_sentence_length = np.max([np.max([len(input_dict['shortest_inputs']['shortest_token']),
                                                       len(input_dict['shortest_inputs']
                                                           ['shortest_pos']),
                                                       len(input_dict['shortest_inputs']
                                                           ['shortest_dist1']),
                                                       len(input_dict['shortest_inputs']['shortest_dist2'])]) for input_dict in test_batch_data])

        for input_dict in test_batch_data:
            inputs = input_dict['full_inputs']
            shortset_inputs = input_dict['shortest_inputs']
            entity_tag = input_dict['entity_pair']
            label = input_dict['label']
            entity_idx_to_type = input_dict['entity_idx_to_type']

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
            dep, dep_idx = prepare_sequence(shortset_inputs['shortest_dep'],
                                            dependency_to_ix, max_length=shortest_max_sentence_length,
                                            prepare_word=False, padding=True)

            shortest_token_batch.append(token_idx)
            shortest_pos_batch.append(pos_idx)
            shortest_dist1_batch.append(dist1_idx)
            shortest_dist2_batch.append(dist2_idx)
            shortest_lengths_batch.append(shortest_lengths)
            label_batch.append(torch.tensor([label], dtype=torch.long))

        inds = np.argsort(-torch.stack(shortest_lengths_batch).numpy())
        shortest_lengths_batch = torch.stack(shortest_lengths_batch)[inds]
        label_batch = torch.stack(label_batch)[inds]

        shortest_batch_in = {
            'token': torch.stack(shortest_token_batch)[inds],
            'pos': torch.stack(shortest_pos_batch)[inds],
            'dist1': torch.stack(shortest_dist1_batch)[inds],
            'dist2': torch.stack(shortest_dist2_batch)[inds]
        }

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(shortest_batch_in, shortest_lengths_batch)

        if len(outputs.size()) == 1:  # batch_size is 1
            outputs = outputs.unsqueeze(0)

        predictions.append(outputs)
        preds = torch.max(outputs, dim=1)[1]

        y_true += [x.item() for x in label_batch]
        y_pred += [x.item() for x in preds]

    return np.array(y_true), np.array(y_pred), predictions
