import torch
from stylenet.modules import FactoredLSTM, Encoder


def test_encoder():
    feature_dim = 256
    batch_size = 5
    channel_size = 3
    img_size = 256

    encoder = Encoder(feature_dim)
    images = torch.randn(batch_size, channel_size, img_size, img_size)

    features = encoder(images)

    assert features.shape == (batch_size, feature_dim)

def test_factored_lstm():
    emb_dim = 3
    vocab_size = 5
    hidden_dim = 7
    style_dim = 11
    num_layers = 13
    random_init = 1.0
    dropout = 0.3

    batch_size = 17
    max_len = 23

    DEFAULT_MODE = 'default'
    STYLE_MODE = 'style'

    rnn_inputs = torch.randint(0, vocab_size, (batch_size, max_len))
    sentence_vector = torch.randn(batch_size, hidden_dim)

    factored_lstm = FactoredLSTM(emb_dim, vocab_size, hidden_dim,
                                 style_dim, num_layers, random_init,
                                 dropout,
                                 )

    default_outputs = factored_lstm.forward(rnn_inputs, sentence_vector, DEFAULT_MODE)
    style_outputs = factored_lstm.forward(rnn_inputs, None, STYLE_MODE)

    assert len(default_outputs) == (batch_size)
    assert default_outputs[0].shape == (max_len-1, vocab_size)

    assert len(style_outputs) == (batch_size)
    assert style_outputs[0].shape == (max_len-1, vocab_size)
