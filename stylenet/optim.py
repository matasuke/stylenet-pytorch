from typing import Optional, List

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

OPT_LIST = ['sgd', 'adagrad', 'adadelta', 'adam']


class Optim:
    '''
    wrapper for optimizer
    '''
    def __init__(
            self,
            method: str,
            lr: float,
            max_grad_norm: Optional[float]=None,
            lr_decay: int=1,
            start_decay_at: Optional[int]=None,
    ):
        '''
        wrap optimizer

        :param method: method to optimize. choose from sgd, adagrad, adadelta, adam.
        :param lr: learning rate
        :param max_grad_norm: maximum gradient norm
        :param lr_decay: decay ratio for learning rate
        :param start_decay_at: start learning rate decay at this epoch
        '''
        assert method in OPT_LIST

        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

    def set_parameters(self, params: List):
        '''
        set parameters to optimizer
        parameters has to be list.

        :param params: parameters to be updated
        '''
        self.params = params

        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise ValueError(f'Unknown optimizer method: {self.method}')

    def step(self):
        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()

    def update_learning_rate(self, ppl: float, epoch: int):
        '''
        decay learning rate if val pref does not improve or we hit the start_decay_at limit

        :param ppl: last perplexity
        :param epoch: current epoch
        '''
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print(f'Decaying learning rate to {self.lr}')

        self.last_ppl = ppl
        self.optimizer.param_groups[0]['lr'] = self.lr
