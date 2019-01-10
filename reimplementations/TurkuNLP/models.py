from imports import *
from constants import *


class Turku(nn.Module):

    def __init__(self, vocab_size, pos_size, dependency_size, pretrained_embedding_matrix, batch_size=1,
                 embedding_dim=WORD_EMBEDDING_DIM, pos_embedding_dim=POS_EMBEDDING_DIM,
                 dependency_embedding_dim=DEPENDENCY_EMBEDDING_DIM, hidden_dim=128, dropout=0.5):
        super(Turku, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(
            torch.from_numpy(pretrained_embedding_matrix))

        self.pos_embeddings = nn.Embedding(pos_size, pos_embedding_dim)

        self.dependency_embeddings = nn.Embedding(
            dependency_size, dependency_embedding_dim)

        self.embedding_dim = embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.dependency_embedding_dim = dependency_embedding_dim

        self.word_lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2, bidirectional=True)
        self.pos_lstm = nn.LSTM(
            pos_embedding_dim, hidden_dim // 2, bidirectional=True)
        self.dep_lstm = nn.LSTM(dependency_embedding_dim,
                                hidden_dim // 2, bidirectional=True)

        self.hidden_1 = nn.Linear((3*hidden_dim)+1, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.hidden2arg = nn.Linear(hidden_dim, 2)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2),
                torch.randn(2, self.batch_size, self.hidden_dim // 2))

    def _get_lstm_out(self, inputs):

        word_embeddings = self.word_embeddings(inputs['token'])
        pos_embeddings = self.pos_embeddings(inputs['pos'])
        dependency_embeddings = self.dependency_embeddings(inputs['dep'])

        batch_size, word_seq_len, _ = word_embeddings.size()
        _, pos_seq_len, _ = pos_embeddings.size()
        _, dep_seq_len, _ = dependency_embeddings.size()

        self.batch_size = batch_size
        self.hidden = self.init_hidden()

        word_lstm_out, _ = self.word_lstm(word_embeddings.view(
            word_seq_len, batch_size, -1), self.hidden)
        pos_lstm_out, _ = self.pos_lstm(pos_embeddings.view(
            pos_seq_len, batch_size, -1), self.hidden)
        dep_lstm_out, _ = self.dep_lstm(dependency_embeddings.view(
            dep_seq_len, batch_size, -1), self.hidden)

        return F.sigmoid(word_lstm_out), F.sigmoid(pos_lstm_out), F.sigmoid(dep_lstm_out)

    def forward(self, inputs, lengths, habitat_geographical_flag):
        word_lstm_out, pos_lstm_out, dep_lstm_out = self._get_lstm_out(inputs)

        feature_vector = torch.cat(
            (word_lstm_out[-1], pos_lstm_out[-1], dep_lstm_out[-1], habitat_geographical_flag), 1)

        hidden_vector = F.sigmoid(self.hidden_1(feature_vector))
        dropout_vector = self.dropout(hidden_vector)
        arg_role_score = F.softmax(self.hidden2arg(dropout_vector))
        return arg_role_score
