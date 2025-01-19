import torch
import torch.nn as nn

from papers.show_attend_tell.models.attention import Attention


class DecoderWithAttention(nn.Module):
    """
    Decoder (LSTM) with soft attention on encoder output.
    """

    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        encoder_dim=2048,
        dropout=0.5,
    ):
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout_rate = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)

        self.dropout = nn.Dropout(p=self.dropout_rate)

        # Linear layer to find scores over vocabulary
        self.fc = nn.Linear(decoder_dim, vocab_size)

        # Initialize LSTM state
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # to initialize hidden state
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # to initialize cell state

        self.init_weights()

    def init_weights(self):
        """
        Initialize some layers with uniform distributions for embeddings and linear layers.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_states(self, encoder_out):
        """
        Average the encoder_out features across all pixels
        and transform to get initial hidden and cell states.
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, captions, lengths):
        """
        Forward pass for training with teacher forcing.

        encoder_out: (batch_size, encoder_dim, num_pixels)
        captions: (batch_size, max_length) ground-truth captions
        lengths: (batch_size,) actual lengths of the captions

        returns:
            predictions (batch_size, max_length, vocab_size),
            alphas (attention weights for each timestep),
            sorted_lengths (useful if you sorted sequences)
        """
        # Transpose encoder_out to (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.permute(0, 2, 1)

        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        max_length = max(lengths)

        # Embedding for captions
        embeddings = self.embedding(captions)  # (batch_size, max_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_states(encoder_out)  # (batch_size, decoder_dim)

        # We'll be storing predictions and attention weights at each step
        predictions = torch.zeros(batch_size, max_length, self.vocab_size).to(
            encoder_out.device
        )
        alphas = torch.zeros(batch_size, max_length, num_pixels).to(encoder_out.device)

        for t in range(max_length - 1): # predict up to last token
            # At each time step t:
            # 1. Compute attention
            context, alpha = self.attention(encoder_out, h)

            # 2. LSTM step
            # use ground-truth word at time t as input
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            h = self.dropout(h)

            # 3. Compute output (vocab scores)
            preds = self.fc(h)  # (batch_size, vocab_size)
            predictions[:, t, :] = preds

            # 4. Save attention weights
            alphas[:, t, :] = alpha

        return predictions, alphas
