import torch.nn as nn


class Attention(nn.Module):
    """
    Soft Attention Network:
        - encoder_out -> linear -> attention space
        - decoder_hidden -> linear -> attention space
        - combine them (ReLU) -> score -> softmax -> alpha (attention weights)
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(
            encoder_dim, attention_dim
        )  # transform encoder's output
        self.decoder_att = nn.Linear(
            decoder_dim, attention_dim
        )  # transform decoder's hidden state
        self.full_att = nn.Linear(attention_dim, 1)  # produce scalar attention score
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax over the spatial locations

    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out: (batch_size, num_pixels, encoder_dim)
        decoder_hidden: (batch_size, decoder_dim)

        returns:
            - attention weighted encoding (batch_size, encoder_dim)
            - attention weights (batch_size, num_pixels)
        """
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att2 = att2.unsqueeze(1)

        att = self.full_att(self.relu(att1 + att2)).squeeze(2)

        alpha = self.softmax(att)

        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, alpha
