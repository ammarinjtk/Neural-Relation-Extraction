from imports import *
from constants import *

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 dropout_p=0.5,
                 h=4,
                 is_masked=False):
        super(MultiHeadAttention, self).__init__()

        if query_dim != key_dim:
            raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        self._key_dim = torch.tensor(
            data=[key_dim], requires_grad=True, dtype=torch.float32).cuda()
        self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)
        self.bn = nn.BatchNorm1d(num_units)

    def forward(self, query, keys):
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)

        # calculate QK^T
        attention = torch.matmul(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)
        attention = attention / torch.sqrt(self._key_dim)
        # use masking (usually for decoder) to prevent leftward
        # information flow and retains auto-regressive property
        # as said in the paper
        if self._is_masked:
            diag_vals = attention[0].sign().abs()
            diag_mat = diag_vals.tril()
            diag_mat = diag_mat.unsqueeze(0).expand(attention.size())
            mask = torch.ones(diag_mat.size()) * (-2**32 + 1)
            # this is some trick that I use to combine the lower diagonal
            # matrix and its masking. (diag_mat-1).abs() will reverse the value
            # inside diag_mat, from 0 to 1 and 1 to zero. with this
            # we don't need loop operation andn could perform our calculation
            # faster
            attention = (attention * diag_mat) + (mask * (diag_mat - 1).abs())
        # put it to softmax
        scores = F.softmax(attention, dim=-1)
        # apply dropout
        attention = F.dropout(scores, self._dropout_p)
        # multiplyt it with V
        attention = torch.matmul(attention, V)
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2)
        # residual connection
        attention += query
        # apply batch normalization
        attention = self.bn(attention.transpose(1, 2)).transpose(1, 2)

        return attention, scores
    
class PositionWiseFFN(nn.Module):
    def __init__(self, feature_size, num_units=[1024, 300]):
        super(PositionWiseFFN, self).__init__()
        self.ffn = self._build_ffn(feature_size, num_units)

    def _build_ffn(self, feature_size, num_units):
        layers = []
        features = feature_size
        for unit in num_units:
            layers.append(nn.Linear(features, unit))
            features = unit

        return nn.Sequential(*layers)

    def forward(self, X):
        # assert if the feature size of inputs not the same as
        # the last ffn layer, since we need both of them
        # the same for residual network
        assert X.size(-1) == self.ffn[-1].bias.size(-1)
        ffn = self.ffn(X)
        # residual network
        ffn += X

        return ffn
    
class SelfAttention(nn.Module):
    """
    attn = SelfAttention(3)
    x = torch.tensor([[[2, 2, 3], [24, 1, 1]], 
                      [[1, 1, 3], [0, 0, 0]], 
                      [[1, 1, 1], [0, 0, 0]]], dtype=torch.float)
    lengths = torch.tensor([2, 1, 1]).float()
    attn(x, lengths)
    """
    def __init__(self, attention_size, non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.attention_weights = Parameter(torch.FloatTensor(attention_size))
        self.softmax = nn.Softmax(dim=-1)

        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = nn.Tanh()

        init.uniform_(self.attention_weights.data, -0.005, 0.005)

    def get_mask(self, attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()

        if attentions.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths.data):  # skip the first sentence
            l = int(l.item())
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, lengths):

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))
        scores = self.softmax(scores)

        # construct a mask, based on the sentence lengths
        # lengths is a 1D Tensor: batch
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states (context vector)
        representations = weighted.sum(1).squeeze()

        return representations, scores
    
class EntityOrientedAttention(nn.Module):
    
    def __init__(self, attention_size, non_linearity="tanh", embedding_size=ELMO_EMBEDDING_DIM):
        super(EntityOrientedAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.embedding_size = embedding_size
        
        self.linear = nn.Linear(attention_size, attention_size)

        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = nn.Tanh()

    def get_mask(self, attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()

        if attentions.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths.data):  # skip the first sentence
            l = int(l.item())
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, entities, lengths):

        # inputs is a 3D Tensor: batch, len, hidden_size
        # entities is 3D Tensor: 2, batch, hidden_size
        a1 = self.softmax(inputs.matmul(entities[0].unsqueeze(2))/self.embedding_size).squeeze(2) # [B, L]
        a2 = self.softmax(inputs.matmul(entities[1].unsqueeze(2))/self.embedding_size).squeeze(2) # [B, L]
        scores = (a1+a2)/2 # [B, L]

        # construct a mask, based on the sentence lengths
        # lengths is a 1D Tensor: batch
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs)) # [B, L, H]
        
        return weighted, scores
    
class CnnText(nn.Module):
    """
    conv = CnnText()
    x = torch.rand(3, 5, 100)
    x.shape
    conv(x)
    """
    def __init__(self, embed_dim=100, num_filters=30, window_sizes=(3, 4, 5), h=8):
        super(CnnText, self).__init__()
        
        self.multihead_attn = MultiHeadAttention(num_filters, num_filters, num_filters, h=h)
        
        self.multihead_attns = nn.ModuleList(
            [MultiHeadAttention(num_filters, num_filters, num_filters, h=h) for size in range(3)])

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, [window_size, embed_dim], padding=(window_size-1, 0))
            for window_size in window_sizes
        ])

    def forward(self, x):
        # input x: [B, T, E]
        # Apply a convolution + max pool layer for each window size
        x = torch.unsqueeze(x, 1)       # [B, C, T, E] Add a channel dim.
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))        # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)            # [B, F, window]

        x = x.view(x.size(0), -1)       # [B, F * window]
        
        return x

    
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

    
class Frankenstein(nn.Module):

    def __init__(self, vocab_size, pos_size, distance_size, max_sequence_length, 
                 pretrained_embedding_matrix, distance_pretrain_embedding_matrix, batch_size=1, 
                 word_embedding_dim=WORD_EMBEDDING_DIM, pos_embedding_dim=POS_EMBEDDING_DIM, 
                 distance_embedding_dim=DISTANCE_EMBEDDING_DIM, 
                 hidden_dim=128, drop=0.5, h=8, multihead_sizes=3):
        super(Frankenstein, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.distance_embedding_dim = distance_embedding_dim
        
        self.all_embedding_dim = ELMO_EMBEDDING_DIM+pos_embedding_dim+(2*distance_embedding_dim)

        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embedding_matrix))
        self.pos_embeddings = nn.Embedding(pos_size, pos_embedding_dim)
        
        self.position_embeddings = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_sequence_length+1, ELMO_EMBEDDING_DIM, padding_idx=0),
            freeze=True)
        
        # Map relative distance (-1) to 50 dimension-vector from look up table
        # Then, scaled by max corpus distance and passed through tanh activation
        self.distance_embeddings = nn.Embedding(distance_size, distance_embedding_dim)
        self.distance_embeddings.weight.data.copy_(torch.from_numpy(distance_pretrain_embedding_matrix))

        self.rnn = nn.LSTM(self.all_embedding_dim, 
                           hidden_dim // 2, 
                           bidirectional=True)
        
        # Attentions
        self.attn = EntityOrientedAttention(word_embedding_dim)
        self.self_attn = SelfAttention(hidden_dim)
        self.multihead_attns = nn.ModuleList(
            [MultiHeadAttention(self.all_embedding_dim, self.all_embedding_dim, self.all_embedding_dim, h=h) for size in range(multihead_sizes)])

        self.dropout = nn.Dropout(drop)

        self.output = nn.Linear(hidden_dim+self.all_embedding_dim+BERT_FEATURE_DIM, 2)
        
        self.rnn_hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.rand(2, self.batch_size, self.hidden_dim // 2).cuda(), 
                torch.rand(2, self.batch_size, self.hidden_dim // 2).cuda())
    
    def _get_embedding(self, inputs, shortest_inputs, entities, lengths, ELMO_embeddings, ELMO_shortest_embeddings, ELMO_entity_embeddings):
        
        # Full sentence
        pos_embeddings = self.pos_embeddings(inputs['pos'])
        dist_1_embeddings = self.distance_embeddings(inputs['dist1'])
        dist_2_embeddings = self.distance_embeddings(inputs['dist2'])
        
        batch_size, seq_len, _ = pos_embeddings.size()
        
        pos_embeddings = pos_embeddings.view(seq_len, batch_size, -1)
        dist_1_embeddings = dist_1_embeddings.view(seq_len, batch_size, -1)
        dist_2_embeddings = dist_2_embeddings.view(seq_len, batch_size, -1)
        
        ## Entity-oriented Attention for ELMo embeddings
        ELMO_entity_embeddings = ELMO_entity_embeddings.view(2, batch_size, -1)
        ELMO_embeddings, _ = self.attn(ELMO_embeddings, ELMO_entity_embeddings, lengths)
        ELMO_embeddings = ELMO_embeddings.view(seq_len, batch_size, -1)

        embedding_vector = F.relu(torch.cat((ELMO_embeddings, pos_embeddings, dist_1_embeddings, dist_2_embeddings), 2))
        
        # SDP
        pos_embeddings = self.pos_embeddings(shortest_inputs['pos'])
        dist_1_embeddings = self.distance_embeddings(shortest_inputs['dist1'])
        dist_2_embeddings = self.distance_embeddings(shortest_inputs['dist2'])
        position_embeddings = self.position_embeddings(shortest_inputs['position'])

        batch_size, seq_len, _ = pos_embeddings.size()

        pos_embeddings = pos_embeddings.view(seq_len, batch_size, -1)
        position_embeddings = position_embeddings.view(seq_len, batch_size, -1)
        dist_1_embeddings = dist_1_embeddings.view(seq_len, batch_size, -1)
        dist_2_embeddings = dist_2_embeddings.view(seq_len, batch_size, -1)
        
        ELMO_shortest_embeddings = ELMO_shortest_embeddings.view(seq_len, batch_size, -1)
        ELMO_shortest_embeddings = ELMO_shortest_embeddings + position_embeddings

        shortest_embedding_vector = F.relu(torch.cat((ELMO_shortest_embeddings, pos_embeddings, dist_1_embeddings, dist_2_embeddings), 2))

        return embedding_vector, shortest_embedding_vector.view(batch_size, seq_len, -1), batch_size
    
    def _get_rnn_out(self, embedding_vector, lengths, batch_size):

        self.batch_size = batch_size
        self.rnn_hidden = self.init_hidden()
        
        pack = torch.nn.utils.rnn.pack_padded_sequence(embedding_vector, lengths, batch_first=False)
        
        rnn_out, _ = self.rnn(pack, self.rnn_hidden)
        
        unpack, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        return unpack

    def _get_cnn_out(self, embedding_vector):
        
        batch_size, seq_len, embedding_dim = embedding_vector.size()
        
        # Multi-Head attention
        decode_vector = embedding_vector
        for i, mhattn in enumerate(self.multihead_attns):
            mhattn_vect, _ = mhattn(decode_vector, decode_vector)
            decode_vector = F.relu(mhattn_vect)
        
        decode_vector = decode_vector.contiguous().view(batch_size, -1, seq_len) # [B, H, F]
        decode_vector = F.max_pool1d(decode_vector, decode_vector.size(2))  # [B, H, 1]
        
        return decode_vector.squeeze(2)

    
    def forward(self, inputs, shortest_inputs, entities, lengths, ELMO_embeddings, 
                ELMO_shortest_embeddings, ELMO_entity_embeddings):

        rnn_embedding_vector, cnn_embedding_vector, batch_size = self._get_embedding(inputs, shortest_inputs, entities, lengths, ELMO_embeddings, ELMO_shortest_embeddings, ELMO_entity_embeddings)

        rnn_out = self._get_rnn_out(rnn_embedding_vector, lengths, batch_size)

        cnn_out = self._get_cnn_out(cnn_embedding_vector)
        
        # Additive attention
        context_vector, _ = self.self_attn(rnn_out, lengths)
        if len(context_vector.size()) == 1:
            context_vector = context_vector.unsqueeze(0)
        
        concat_vector = F.relu(torch.cat((context_vector, cnn_out, inputs['bert_features']), 1))
        
        concat_vector = self.dropout(concat_vector)
        output_vector = self.output(concat_vector)
        
        return F.softmax(output_vector, dim=1)