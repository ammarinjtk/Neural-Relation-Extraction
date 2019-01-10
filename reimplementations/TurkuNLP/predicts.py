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
    
    
    for i in range(0,len(test_data), batch_size):
        test_batch_data = np.array(test_data)[i:i+batch_size]
        
        entity_type_batch = []

        shortest_token_batch = []
        shortest_pos_batch = []
        shortest_dep_batch = []
        shortest_position_batch = []
        shortest_lengths_batch = []

        label_batch = []

        shortest_max_sentence_length = np.max([np.max([len(input_dict['shortest_inputs']['shortest_token']), 
                                                       len(input_dict['shortest_inputs']['shortest_pos']), 
                                                       len(input_dict['shortest_inputs']['shortest_dep'])]) for input_dict in test_batch_data])

        for input_dict in test_batch_data:
            inputs = input_dict['full_inputs']
            shortset_inputs = input_dict['shortest_inputs']
            entity_tag = input_dict['entity_pair']
            label = input_dict['label']
            entity_idx_to_type = input_dict['entity_idx_to_type']

            label_batch.append(torch.tensor([label], dtype=torch.long))


            token, token_idx, shortest_lengths = prepare_sequence(shortset_inputs['shortest_token'], 
                                                word_to_ix, max_length=shortest_max_sentence_length, padding=True, get_lengths=True)
            pos, pos_idx = prepare_sequence(shortset_inputs['shortest_pos'], 
                                            pos_to_ix, max_length=shortest_max_sentence_length, 
                                            prepare_word=False, padding=True)
            dep, dep_idx = prepare_sequence(shortset_inputs['shortest_dep'], 
                                            dependency_to_ix, max_length=shortest_max_sentence_length, 
                                            prepare_word=False, padding=True)

            shortest_token_batch.append(token_idx)
            shortest_pos_batch.append(pos_idx)
            shortest_dep_batch.append(dep_idx)
            shortest_lengths_batch.append(shortest_lengths)
            
            corresponding_entity_type = torch.tensor([1], dtype=torch.float) if entity_idx_to_type[entity_tag[1]] == 'Habitat' else torch.tensor([0], dtype=torch.float)
            entity_type_batch.append(corresponding_entity_type)

        # Sort lengths in decending order
        inds = np.argsort(-torch.stack(shortest_lengths_batch).numpy())   
        label_batch = torch.stack(label_batch)[inds]
        shortest_lengths_batch = torch.stack(shortest_lengths_batch)[inds]
        entity_type_batch = torch.stack(entity_type_batch)[inds]

        shortest_batch_in = {
            'token': torch.stack(shortest_token_batch)[inds],
            'pos': torch.stack(shortest_pos_batch)[inds],
            'dep': torch.stack(shortest_dep_batch)[inds],
        }
        
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(shortest_batch_in, shortest_lengths_batch, entity_type_batch)
        
        if len(outputs.size()) == 1: # batch_size is 1
            outputs = outputs.unsqueeze(0)
        
        predictions.append(outputs)
        preds = torch.max(outputs, dim=1)[1]
        
        y_true += [x.item() for x in label_batch]
        y_pred += [x.item() for x in preds]

    return np.array(y_true), np.array(y_pred), predictions
