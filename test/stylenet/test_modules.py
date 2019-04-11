import torch
from stylenet.modules import FactoredLSTM, EncoderCNN


def test_encoder():
    feature_dim = 256
    batch_size = 5
    channel_size = 3
    img_size = 256

    encoder = EncoderCNN(feature_dim)
    images = torch.randn(batch_size, channel_size, img_size, img_size)

    features = encoder(images)

    assert features.shape == (batch_size, feature_dim)


class TestFactoredLSTM:
    '''
    class for testing FactoredLSTM
    '''
    def test_intput_feed(self):
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

        factored_lstm = FactoredLSTM(
            emb_dim, vocab_size, hidden_dim,
            style_dim, num_layers, random_init,
            dropout,
        )

        default_outputs = factored_lstm.forward(rnn_inputs, sentence_vector, DEFAULT_MODE)
        style_outputs = factored_lstm.forward(rnn_inputs, None, STYLE_MODE)

        assert default_outputs.shape == (batch_size, max_len, vocab_size)
        assert style_outputs.shape == (batch_size, max_len, vocab_size)

    def test_backward(self):
        emb_dim = 3
        vocab_size = 5
        hidden_dim = 7
        style_dim = 11
        num_layers = 1
        random_init = 1.0
        dropout = 0
        batch_size = 5
        iterations = 1000

        DEVICE = 'cuda:3'
        DEFAULT_MODE = 'default'
        STYLE_MODE = 'style'

        rnn_inputs = torch.LongTensor([
            [1, 4, 3, 3, 2, 0, 0],
            [1, 3, 3, 4, 4, 0, 0],
            [1, 3, 3, 4, 4, 0, 0],
            [1, 3, 3, 4, 4, 0, 0],
            [1, 3, 3, 4, 4, 0, 0],
        ]).to(DEVICE)
        targets = torch.LongTensor([
            [1, 4, 3, 3, 2, 0, 0],
            [1, 3, 3, 4, 4, 0, 0],
            [1, 3, 3, 4, 4, 0, 0],
            [1, 3, 3, 4, 4, 0, 0],
            [1, 3, 3, 4, 4, 0, 0],
        ]).to(DEVICE)
        sentence_vector = torch.randn(batch_size, hidden_dim).to(DEVICE)

        factored_lstm = FactoredLSTM(
            emb_dim, vocab_size, hidden_dim,
            style_dim, num_layers, random_init,
            dropout,
        )
        factored_lstm = factored_lstm.to(DEVICE)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        normal_params = list(factored_lstm.default_parameters())
        style_params = list(factored_lstm.style_parameters())
        normal_optimizer = torch.optim.Adam(normal_params, lr=0.001)
        style_optimizer = torch.optim.Adam(style_params, lr=0.001)

        for _ in range(iterations):
            default_outputs = factored_lstm.forward(rnn_inputs, sentence_vector, DEFAULT_MODE)
            default_loss = criterion(default_outputs.view(-1, default_outputs.size(-1)), targets.view(-1))
            factored_lstm.zero_grad()
            default_loss.backward()
            normal_optimizer.step()

            sentence_vector = factored_lstm.generate_random_noise(batch_size, DEVICE)
            style_outputs = factored_lstm.forward(rnn_inputs, sentence_vector, STYLE_MODE)
            style_loss = criterion(style_outputs.view(-1, default_outputs.size(-1)), targets.view(-1))
            factored_lstm.zero_grad()
            style_loss.backward()
            style_optimizer.step()

        assert default_loss < 1e-1
        assert style_loss < 1e-1
