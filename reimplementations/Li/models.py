from imports import *
from constants import *


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


class Li_LSTM_Attention(nn.Module):

    def __init__(self, vocab_size, pos_size, distance_size, pretrained_embedding_matrix,
                 distance_pretrain_embedding_matrix, batch_size=1,
                 embedding_dim=WORD_EMBEDDING_DIM, pos_embedding_dim=POS_EMBEDDING_DIM,
                 distance_embedding_dim=DISTANCE_EMBEDDING_DIM, hidden_dim=64, dropout=0.5, num_layer=1):
        super(Li_LSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layer = num_layer

        self.embedding_dim = embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.distance_embedding_dim = distance_embedding_dim

        self.all_embedding_dim = embedding_dim + \
            pos_embedding_dim+(2*distance_embedding_dim)

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(
            torch.from_numpy(pretrained_embedding_matrix))

        self.pos_embeddings = nn.Embedding(pos_size, pos_embedding_dim)

        self.distance_embeddings = nn.Embedding(
            distance_size, distance_embedding_dim)
        self.distance_embeddings.weight.data.copy_(
            torch.from_numpy(distance_pretrain_embedding_matrix))

        self.rnn = nn.GRU(self.all_embedding_dim, hidden_dim //
                          2, num_layer, bidirectional=True)

        self.self_attn = SelfAttention(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.hidden2arg = nn.Linear(hidden_dim, 2)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.randn(2*self.num_layer, self.batch_size, self.hidden_dim // 2)

    def _get_embedding(self, shortest_inputs):

        # Shortest
        word_embeddings = self.word_embeddings(shortest_inputs['token'])
        pos_embeddings = self.pos_embeddings(shortest_inputs['pos'])
        dist_1_embeddings = self.distance_embeddings(shortest_inputs['dist1'])
        dist_2_embeddings = self.distance_embeddings(shortest_inputs['dist2'])

        batch_size, seq_len, _ = word_embeddings.size()

        word_embeddings = word_embeddings.view(seq_len, batch_size, -1)
        pos_embeddings = pos_embeddings.view(seq_len, batch_size, -1)
        dist_1_embeddings = dist_1_embeddings.view(seq_len, batch_size, -1)
        dist_2_embeddings = dist_2_embeddings.view(seq_len, batch_size, -1)

        shortest_embedding_vector = torch.cat(
            (word_embeddings, pos_embeddings, dist_1_embeddings, dist_2_embeddings), 2)

        return shortest_embedding_vector, batch_size

    def _get_rnn_out(self, embedding_vector, shortest_lengths, batch_size):

        self.batch_size = batch_size
        self.rnn_hidden = self.init_hidden()

        pack = torch.nn.utils.rnn.pack_padded_sequence(
            embedding_vector, shortest_lengths, batch_first=False)

        rnn_out, _ = self.rnn(pack, self.rnn_hidden)

        unpack, _ = torch.nn.utils.rnn.pad_packed_sequence(
            rnn_out, batch_first=True)

        return unpack

    def forward(self, shortest_inputs, shortest_lengths):

        embedding_vector, batch_size = self._get_embedding(shortest_inputs)

        rnn_out = self._get_rnn_out(
            embedding_vector, shortest_lengths, batch_size)

        rnn_out = F.tanh(rnn_out)
        context_vector, _ = self.self_attn(rnn_out, shortest_lengths)
        if len(context_vector.size()) == 1:
            context_vector = context_vector.unsqueeze(0)
        context_vector = F.tanh(context_vector)
        dropout_vector = self.dropout(context_vector)
        arg_role_score = F.softmax(self.hidden2arg(dropout_vector))

        return arg_role_score
