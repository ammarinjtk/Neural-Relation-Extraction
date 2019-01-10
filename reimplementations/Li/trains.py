from imports import *
from constants import *
from utils import *

def train_model(model, trn_data, val_data, word_to_ix, pos_to_ix, 
                distance_to_ix, dependency_to_ix, criterion, optimizer, 
                lr_scheduler, num_epochs=4, early_stopped_patience=30, 
                batch_size=3, verbose=True):
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = -1
    best_train_f1 = -1
    history = []
    is_early_stopped = False
    early_stopped_patience_count = 1
    
    trn_data_len = len(trn_data)
    val_data_len = len(val_data)

    for epoch in range(num_epochs):
        
        if early_stopped_patience > 0 and is_early_stopped:
            if verbose:
                print("Early stopping, val F1_score at {:.4f}".format(best_val_f1))
            break
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        model.train()  # Set model to training mode
        
        batch_count = 1
        batch_loss = 0.0
        batch_f1 = 0.0
        
        # Iterate over data.
        for i in range(0,trn_data_len, batch_size):
            trn_batch_data = np.array(trn_data)[i:i+batch_size]

            shortest_token_batch = []
            shortest_pos_batch = []
            shortest_dist1_batch = []
            shortest_dist2_batch = []
            shortest_lengths_batch = []

            label_batch = []

            shortest_max_sentence_length = np.max([np.max([len(input_dict['shortest_inputs']['shortest_token']), 
                                                           len(input_dict['shortest_inputs']['shortest_pos']), 
                                                           len(input_dict['shortest_inputs']['shortest_dist1']),
                                                           len(input_dict['shortest_inputs']['shortest_dist2'])]) for input_dict in trn_batch_data])
            

            for input_dict in trn_batch_data:
                
                inputs = input_dict['full_inputs']
                shortset_inputs = input_dict['shortest_inputs']
                entity_tag = input_dict['entity_pair']
                label = input_dict['label']
                entity_idx_to_type = input_dict['entity_idx_to_type']

                token, token_idx, shortest_lengths = prepare_sequence(shortset_inputs['shortest_token'], word_to_ix, max_length=shortest_max_sentence_length, padding=True, get_lengths=True)
                pos, pos_idx = prepare_sequence(shortset_inputs['shortest_pos'], pos_to_ix, max_length=shortest_max_sentence_length, prepare_word=False, padding=True)
                dist1, dist1_idx = prepare_sequence(shortset_inputs['shortest_dist1'], distance_to_ix, max_length=shortest_max_sentence_length, prepare_word=False, padding=True)
                dist2, dist2_idx = prepare_sequence(shortset_inputs['shortest_dist2'], distance_to_ix, max_length=shortest_max_sentence_length, prepare_word=False, padding=True)

                shortest_token_batch.append(token_idx)
                shortest_pos_batch.append(pos_idx)
                shortest_dist1_batch.append(dist1_idx)
                shortest_dist2_batch.append(dist2_idx)
                shortest_lengths_batch.append(shortest_lengths)
                label_batch.append(torch.tensor([label], dtype=torch.long))

            # Sort lengths in decending order
            inds = np.argsort(-torch.stack(shortest_lengths_batch).numpy())
            shortest_lengths_batch = torch.stack(shortest_lengths_batch)[inds]
            label_batch = torch.stack(label_batch)[inds]

            shortest_batch_in = {
                'token': torch.stack(shortest_token_batch)[inds],
                'pos': torch.stack(shortest_pos_batch)[inds],
                'dist1': torch.stack(shortest_dist1_batch)[inds],
                'dist2': torch.stack(shortest_dist2_batch)[inds],
            }

            # zero the parameter gradients
            optimizer.zero_grad()
            
            outputs = model(shortest_batch_in, shortest_lengths_batch)
            
            if len(outputs.size()) == 1: # batch_size is 1
                outputs = outputs.unsqueeze(0)

            loss = criterion(outputs, label_batch.squeeze(1))
            preds = torch.max(outputs, dim=1)[1]

            running_f1 = f1_score(np.array([x.item() for x in label_batch]), 
                                np.array(preds))

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            batch_loss += loss.item()
            batch_f1 += running_f1
            batch_count += 1

        epoch_loss = batch_loss / batch_count
        epoch_f1 = batch_f1 / batch_count
        

        batch_count = 1
        batch_loss = 0.0
        batch_f1 = 0.0
        
        if val_data is not None:
            
            model.eval()   # Set model to evaluate mode
            
            # Iterate over data.
            for i in range(0,val_data_len, batch_size):
                val_batch_data = np.array(val_data)[i:i+batch_size]
                shortest_token_batch = []
                shortest_pos_batch = []
                shortest_dist1_batch = []
                shortest_dist2_batch = []
                shortest_lengths_batch = []

                label_batch = []

                shortest_max_sentence_length = np.max([np.max([len(input_dict['shortest_inputs']['shortest_token']), 
                                                           len(input_dict['shortest_inputs']['shortest_pos']), 
                                                           len(input_dict['shortest_inputs']['shortest_dist1']),
                                                           len(input_dict['shortest_inputs']['shortest_dist2'])]) for input_dict in val_batch_data])

                for input_dict in val_batch_data:
                    inputs = input_dict['full_inputs']
                    shortset_inputs = input_dict['shortest_inputs']
                    entity_tag = input_dict['entity_pair']
                    label = input_dict['label']
                    entity_idx_to_type = input_dict['entity_idx_to_type']

                    token, token_idx, shortest_lengths = prepare_sequence(shortset_inputs['shortest_token'], word_to_ix, max_length=shortest_max_sentence_length, padding=True, get_lengths=True)
                    pos, pos_idx = prepare_sequence(shortset_inputs['shortest_pos'], pos_to_ix, max_length=shortest_max_sentence_length, prepare_word=False, padding=True)
                    dist1, dist1_idx = prepare_sequence(shortset_inputs['shortest_dist1'], distance_to_ix, max_length=shortest_max_sentence_length, prepare_word=False, padding=True)
                    dist2, dist2_idx = prepare_sequence(shortset_inputs['shortest_dist2'], distance_to_ix, max_length=shortest_max_sentence_length, prepare_word=False, padding=True)
                    
                    shortest_token_batch.append(token_idx)
                    shortest_pos_batch.append(pos_idx)
                    shortest_dist1_batch.append(dist1_idx)
                    shortest_dist2_batch.append(dist2_idx)
                    shortest_lengths_batch.append(shortest_lengths)
                    label_batch.append(torch.tensor([label], dtype=torch.long))

                inds = np.argsort(-torch.stack(shortest_lengths_batch).numpy())
                label_batch = torch.stack(label_batch)[inds]
                shortest_lengths_batch = torch.stack(shortest_lengths_batch)[inds]
                

                shortest_batch_in = {
                    'token': torch.stack(shortest_token_batch)[inds],
                    'pos': torch.stack(shortest_pos_batch)[inds],
                    'dist1': torch.stack(shortest_dist1_batch)[inds],
                    'dist2': torch.stack(shortest_dist2_batch)[inds]
                }

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(shortest_batch_in, shortest_lengths_batch)
                  
                if len(outputs.size()) == 1: # batch_size is 1
                    outputs = outputs.unsqueeze(0)
                
                loss = criterion(outputs, label_batch.squeeze(1))
                preds = torch.max(outputs, dim=1)[1]

                running_f1 = f1_score(np.array([x.item() for x in label_batch]), 
                                    np.array(preds))

                # statistics
                batch_loss += loss.item()
                batch_f1 += running_f1
                batch_count += 1

            val_epoch_loss = batch_loss / batch_count
            val_epoch_f1 = batch_f1 / batch_count
            
            if lr_scheduler is not None:
                lr_scheduler.step(val_epoch_f1)
            if verbose:
                print('Train, (Loss: {:.4f}, F1_score: {:.4f}), Val, (Loss: {:.4f}, F1_score: {:.4f})'.format(epoch_loss, epoch_f1, val_epoch_loss, val_epoch_f1))
            
            history.append({
                'loss': epoch_loss,
                'val_loss': val_epoch_loss,
                'f1': epoch_f1,
                'val_f1': val_epoch_f1
            })
            if early_stopped_patience > 0:
                if val_epoch_f1 > best_val_f1:
                    # improve
                    if verbose:
                        print("Val F1_score improved from {:.4f} to {:.4f}".format(best_val_f1, val_epoch_f1))
                    if lr_scheduler is not None:
                        lr_scheduler.cooldown_counter = 0
                        lr_scheduler.num_bad_epochs = 0
                    best_val_f1 = val_epoch_f1
                    best_train_f1 = epoch_f1
                    early_stopped_patience_count = 1
                    best_model_wts = copy.deepcopy(model.state_dict())
                elif best_val_f1 == 0:
                    # best f1_score still 0
                    if verbose:
                        print("Val F1_score did not improve from {:.4f}".format(val_epoch_f1))
                    early_stopped_patience_count = 1
                elif early_stopped_patience_count < early_stopped_patience:
                    if verbose:
                        print("Val F1_score did not improve from {:.4f}, early_stopped_patience is {}/{}".format(best_val_f1, early_stopped_patience_count, early_stopped_patience))
                    early_stopped_patience_count += 1
                else:
                    # do not improve
                    if verbose:
                        print("Val F1_score did not improve from {:.4f}".format(best_val_f1))
                    is_early_stopped = True
                    break
            else:
                # do not use early stopping, keep every epoch the best model
                best_model_wts = copy.deepcopy(model.state_dict())
                best_val_f1 = val_epoch_f1
                best_train_f1 = epoch_f1
                
            if verbose:
                print()
        else:
            # no validation
            if verbose:
                print('Train, (Loss: {:.4f}, F1_score: {:.4f})'.format(epoch_loss, epoch_f1))
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    if verbose:
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return (model, best_train_f1, best_val_f1, history)
