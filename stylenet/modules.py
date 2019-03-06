from typing import Optional, Tuple
import warnings
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Encoder(nn.Module):
    '''
    resnet based Encoder to extract image features.
    '''
    __slots__ = ['resnet']

    def __init__(self, feature_dim: int, fix_weights: bool=False):
        '''
        :param feature_dim: dimention of image features.
        :param fix_weights: fix weights of resnet.
        '''
        super(Encoder, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        if fix_weights:
            for param in self.reset.parameters():
                param.requires_grad = False

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, feature_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        '''
        extract image features from images.

        :param images: images for extracting features
        '''
        features = self.resnet(images)

        return features


class FactoredLSTM(nn.Module):
    '''
    Factored LSTM used in stylenet.
    '''
    def __init__(self, emb_dim: int, vocab_size: int, hidden_dim: int,
                 style_dim: int, num_layers: int=1, random_init: float=1.0,
                 dropout: float=0,
                 ):
        '''
        :param input_size: input_size.
        '''
        super(FactoredLSTM, self).__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.style_dim = style_dim
        self.num_layers = num_layers
        self.random_init = random_init
        self.dropout = dropout

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layers, so non-zero dropout expects "
                          f"num_layers greater than 1, but got dropout {dropout} and "
                          "num_layers={num_layers}")

        # embedding
        self.embed = nn.Embedding(vocab_size, emb_dim)

        # factored lstm
        self._all_weights = []
        for layer in range(num_layers):
            layer_input_dim = emb_dim if layer == 0 else hidden_dim

            # factored lstm weights
            U = nn.Linear(layer_input_dim, style_dim * 4)
            S1 = nn.Linear(style_dim * 4, style_dim)
            S2 = nn.Linear(style_dim * 4, style_dim)
            V = nn.Linear(style_dim, hidden_dim * 4)
            W = nn.Linear(hidden_dim, hidden_dim * 4)

            factored_lstm_params = {
                f'U_{layer}': U,
                f'V_{layer}': V,
                f'W_{layer}': W,
                f'S1_{layer}': S1,
                f'S2_{layer}': S2
            }
            for param_name, param in factored_lstm_params.items():
                setattr(self, param_name, param)

            self._all_weights.append(factored_lstm_params)

        # projection
        self.projection = nn.Linear(hidden_dim, vocab_size)

    def forward_step(
            self,
            embedded: torch.Tensor,
            h_0: Tuple[torch.Tensor, ...],
            c_0: Tuple[torch.Tensor, ...],
            mode: str='default'
    ) -> Tuple[torch.Tensor, ...]:
        '''
        forward step.

        :param embedded: embedded matrix shape of [batch_size, max_len, emb_dim]
        :param h_0: hidden state of factoredLSTM shape of Tuple[[batch_size, hidden_dim], ...]
        :param c_0: cell state of factoredLSTM shape of Tuple[[batch_size, hidden_dim], ...]
        :param mode: mode to be used.
        '''
        input_vec = embedded
        h_t = h_0
        c_t = c_0

        for idx, weight in enumerate(self._all_weights):
            U = weight['U_' + str(idx)](input_vec)
            if mode == 'default':
                S = weight['S1_' + str(idx)](U)
            elif mode == 'style':
                S = weight['S2_' + str(idx)](U)
            else:
                raise ValueError(f'Unknown mode {mode}')

            V = weight['V_' + str(idx)](S)
            W = weight['W_' + str(idx)](h_t[idx])

            i, f, o, c = torch.split(V, self.hidden_dim, dim=1)
            i_w, f_w, o_w, c_w = torch.split(W, self.hidden_dim, dim=1)

            i_t = F.sigmoid(i + i_w)
            f_t = F.sigmoid(f + f_w)
            o_t = F.sigmoid(o + o_w)
            c_tilda = F.tanh(c + c_w)

            c_t[idx] = f_t * c_t[idx] + i_t * c_tilda
            h_t[idx] = o_t * c_t[idx]
            input_vec = h_t[idx]

        outputs = self.projection(h_t[-1])

        return outputs, h_t, c_t

    def forward(
            self,
            rnn_inputs: torch.Tensor,
            sentence_vector: Optional[torch.Tensor]=None,
            mode: str='default',
    ) -> torch.Tensor:
        '''
        forward pass for factoredLSTM.

        :param rnn_inputs: tensors of padded tokens size of [batch_size, max_len]
        :param sentence_vector: outputs of encoder parts as init state of factored LSTM.
        :param mode: use sentence_vector or not. if false, it uses random noises as inputs.
        :param random_init: init factoredLSTM with uniform distribution.

        :NOTE:
            during training, mode=default with random_init=False, or
            mode=style with random_init=True,
            during inference, mode=default or style and random_init=False.
        '''
        batch_size = rnn_inputs.size(0)

        assert mode in ['default', 'style']
        assert sentence_vector is None or sentence_vector.shape == (batch_size, self.hidden_dim)

        embedded = self.embed(rnn_inputs)  # [batch_size, max_len, emb_dim]

        if sentence_vector is not None:
            h_t = [sentence_vector for _ in range(self.num_layers)]
            c_t = [sentence_vector for _ in range(self.num_layers)]
        else:
            h_t = torch.empty(batch_size, self.hidden_dim)
            c_t = torch.empty(batch_size, self.hidden_dim)
            nn.init.uniform_(h_t, -self.random_init, self.random_init)
            nn.init.uniform_(c_t, -self.random_init, self.random_init)
            h_t = [h_t for _ in range(self.num_layers)]
            c_t = [c_t for _ in range(self.num_layers)]

        # if torch.cuda.is_available():
        #    h_t = [h.cuda() for h in h_t]
        #    c_t = [c.cuda() for c in c_t]

        all_outputs = []
        for idx in range(embedded.size(1) - 1):
            emb = embedded[:, idx, :]
            outputs, h_t, c_t = self.forward_step(emb, h_t, c_t, mode)
            all_outputs.append(outputs)

        all_outputs = torch.stack(all_outputs, dim=1)

        return all_outputs
