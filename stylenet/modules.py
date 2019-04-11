from typing import Tuple, List, Dict, Optional
import warnings
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from stylenet.beam import Beam


class EncoderCNN(nn.Module):
    '''
    resnet based Encoder to extract image features.
    '''
    __slots__ = ['resnet']

    def __init__(self, feature_dim: int):
        '''
        Load the pretrained ResNet-152 and replace fc layer

        :param feature_dim: dimention of image features.
        '''
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim, momentum=0.01)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        '''
        extract image features from images.

        :param images: images for extracting features
        '''
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))

        return features

    def parameters(self):
        return list(self.linear.parameters()) + list(self.bn.parameters())


class FactoredLSTM(nn.Module):
    '''
    Factored LSTM used in stylenet.
    '''
    def __init__(self, emb_dim: int, vocab_size: int, hidden_dim: int,
                 style_dim: int, num_layers: int=1, random_init: float=1.0,
                 dropout_ratio: float=0, train: bool=False, device: str='cpu',
                 ) -> None:
        '''
        :param emb_dim: dimention of embedding layer
        :param vocab_size: vocaburaly size
        :param hidden_dim: dimention of hidden layer
        :param style_dim: dimention of style factor
        :param num_layers: the number of layer
        :param random_init: init value used for random uniform in sentence_vector
        :param dropout_ratio: dropout ratio
        :param train: train or inference
        :param device: device string
        '''
        super(FactoredLSTM, self).__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.style_dim = style_dim
        self.num_layers = num_layers
        self.random_init = random_init
        self.dropout_ratio = dropout_ratio
        self.train = train
        self.device = device

        if not isinstance(dropout_ratio, numbers.Number) or not 0 <= dropout_ratio <= 1 or \
                isinstance(dropout_ratio, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being zeroed")
        if dropout_ratio > 0 and num_layers == 1 and train:
            warnings.warn("dropout_ratio option adds dropout after all but last "
                          "recurrent layers, so non-zero dropout expects "
                          f"num_layers greater than 1, but got dropout {dropout_ratio} and "
                          f"num_layers={num_layers}")

        # embedding
        self.embed = nn.Embedding(vocab_size, emb_dim)

        # factored lstm
        self._all_weights: List[Dict[str, torch.Tensor]] = []
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
                f'S2_{layer}': S2,
            }

            if dropout_ratio > 0 and num_layers > 1 and train:
                dropout = nn.Dropout(dropout_ratio)
                factored_lstm_params[f'dropout_{layer}'] = dropout

            for param_name, param in factored_lstm_params.items():
                setattr(self, param_name, param)

            self._all_weights.append(factored_lstm_params)

        # projection
        self.projection = nn.Linear(hidden_dim, vocab_size)

    def forward_step(
            self,
            embedded: torch.Tensor,
            h_0: torch.Tensor,
            c_0: torch.Tensor,
            mode: str='default',
    ) -> Tuple[torch.Tensor, ...]:
        '''
        forward step.

        :param embedded: embedded matrix shape of [batch_size, max_len, emb_dim]
        :param h_0: hidden state of factoredLSTM shape of Tuple[[batch_size, hidden_dim], ...]
        :param c_0: cell state of factoredLSTM shape of Tuple[[batch_size, hidden_dim], ...]
        :param mode: mode to be used.
        '''
        input_vec = embedded
        h_t = h_0.clone()
        c_t = c_0.clone()

        for idx, weight in enumerate(self._all_weights):
            U = weight['U_' + str(idx)](input_vec)
            if mode == 'default':
                S = weight['S1_' + str(idx)](U)
            elif mode == 'style':
                S = weight['S2_' + str(idx)](U)
            else:
                raise ValueError(f'Unknown mode {mode}')

            V = weight['V_' + str(idx)](S)
            W = weight['W_' + str(idx)](h_0[idx])

            if self.num_layers > 1 and self.train:
                V = weight['dropout_' + str(idx)](V)

            # print(V.shape)
            i, f, o, c = torch.split(V, self.hidden_dim, dim=1)
            i_w, f_w, o_w, c_w = torch.split(W, self.hidden_dim, dim=1)

            i_t = torch.sigmoid(i + i_w)
            f_t = torch.sigmoid(f + f_w)
            o_t = torch.sigmoid(o + o_w)
            c_tilda = torch.tanh(c + c_w)

            c_t[idx] = f_t * c_0[idx] + i_t * c_tilda
            h_t[idx] = o_t * c_t[idx]
            input_vec = h_0[idx]

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
        :param mode: use sentence_vector or not. if false, it uses random noises as inputs.
        :param sentence_vector: outputs of encoder parts as init state of factored LSTM.

        :NOTE:
            during training, set sentence_vector is not None with mode=default,
            or sentence_vector is None with mode=style.
        '''
        batch_size = rnn_inputs.size(0)

        assert mode in ['default', 'style']
        assert sentence_vector is None or sentence_vector.shape == (batch_size, self.hidden_dim)
        assert (sentence_vector is not None and mode == 'default') \
            or (sentence_vector is None and mode == 'style')

        # [batch_size, max_len] -> [batch_size, max_len, emb_dim]
        embedded = self.embed(rnn_inputs)

        if sentence_vector is None and mode == 'style':
            sentence_vector = torch.empty(batch_size, self.hidden_dim).to(self.device)
            nn.init.uniform_(sentence_vector, -self.random_init, self.random_init)

        # prepare sentence vector of num_layers
        h_t = torch.stack([sentence_vector for _ in range(self.num_layers)])
        c_t = torch.stack([sentence_vector for _ in range(self.num_layers)])

        all_outputs = []
        for idx in range(embedded.size(1)):
            emb = embedded[:, idx, :]  # [batch_size, emb_dim]
            outputs, h_t, c_t = self.forward_step(emb, h_t, c_t, mode)
            all_outputs.append(outputs)

        # List[torch.Tensor[batch_size, vocab_size]] -> torch.Tensor[batch_size, max_len, vocab_size]
        all_outputs = torch.stack(all_outputs, dim=1)

        return all_outputs

    def sample_beam(
            self,
            sentence_vector: torch.Tensor,
            beam_width: int=10,
            max_length: int=50,
            mode: str='default',
            sos_id: int=1,
            eos_id: int=2,
    ) -> torch.Tensor:
        '''
        Generate captions for given image features using beam search.

        :param sentence_vector: image featues.
        :param beam_width: beam width
        :param max_length: maximum number of tokens to be generated
        :param mode: mode to be used for generating captions.
        :param sos_id: SOS_ID
        :param eos_id: EOS_ID
        '''
        assert mode in ['default', 'style']

        h_t = torch.stack([sentence_vector for _ in range(self.num_layers)])
        c_t = torch.stack([sentence_vector for _ in range(self.num_layers)])

        sos_id_tensor = torch.LongTensor([sos_id]).to(self.device)
        candidates = [[0, sos_id_tensor, h_t, c_t, [sos_id]]]
        # beam search
        step = 0
        while step < max_length - 1:
            step += 1
            tmp_candidates = []
            is_eos = True
            for score, last_id, h_t, c_t, id_seq in candidates:
                if id_seq[-1] == eos_id:
                    tmp_candidates.append([score, last_id, h_t, c_t, id_seq])
                else:
                    is_eos = False
                    emb = self.embed(last_id)
                    output, h_t, c_t = self.forward_step(emb, h_t, c_t, mode)
                    output = F.log_softmax(output, dim=-1)
                    output, indices = torch.sort(output, descending=True)
                    output = output[:beam_width]
                    indices = indices[:beam_width]
                    score_list = score + output
                    for score, word_id in zip(score_list[0], indices[0]):
                        tmp_candidates.append(
                            [score, word_id.unsqueeze(0), h_t, c_t, id_seq + [int(word_id)]]
                        )

            if is_eos:
                break
            # sort by log probability and get get high prob candidates size of beam_size
            candidates = sorted(tmp_candidates, key=lambda x: -int(x[0])/len(x[-1]))[:beam_width]

        return candidates[0][-1]

    def greedy_search(
            self,
            sentence_vector: torch.Tensor,
            beam_width: int=5,
            max_length: int=50,
            mode: str='default',
            text_preprocessor: TextPreprocessor,
    ) -> torch.Tensor:
        '''
        '''
        assert mode in ['default', 'style']

        sentence_vector = sentence_vector.repeat(1, beam_width, 1)

        batch_size = sentence_vector.size(0)
        h_t = torch.stack([sentence_vector for _ in range(self.num_layers)])
        c_t = torch.stack([sentence_vector for _ in range(self.num_layers)])

        beam = [Beam(beam_width, text_preprocessor, self.device) for _ in range(batch_size)]

        for i in range(max_length):
            next_id = torch.stack(
                [b.get_current_state() for b in beam if not b.done]
            ).t().contiguous().view(-1, 1)

            emb = self.embed(next_id)
            output, h_t, c_t = self.forward_step(emb, h_t, c_t)
            output = F.log_softmax(output, dim=-1)

    def default_parameters(self):
        return list(self.parameters())

    def style_parameters(self):
        style_params = []
        for idx, weight in enumerate(self._all_weights):
            style_params += list(weight['S2_' + str(idx)].parameters())

        return style_params
