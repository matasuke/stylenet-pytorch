from argparse import ArgumentParser
import collections
from pathlib import Path
import pickle
from typing import List, Sequence, Mapping, Optional, Union
import re

from gensim.models import Word2Vec
import numpy as np


DEFAULT_EMBED_DIM = 512
DEFAULT_W2V_WINDOW_SIZE = 5


class TextPreprocessor:
    '''
    Class for preprocessing text data.
    '''
    PAD_ID = 0  # padding
    SOS_ID = 1  # start of sentence
    EOS_ID = 2  # end of sentence
    UNK_ID = 3  # unknown word
    START_ID = 4  # start of word id

    PAD_SYMBOL = '<PAD>'
    SOS_SYMBOL = '<SOS>'
    EOS_SYMBOL = '<EOS>'
    UNK_SYMBOL = '<UNK>'

    IDS = [PAD_ID, SOS_ID, EOS_ID, UNK_ID]
    SYMBOLS = [PAD_SYMBOL, SOS_SYMBOL, EOS_SYMBOL, UNK_SYMBOL]

    def __init__(
            self,
            token2index: Mapping[str, int],
            index2token: Mapping[int, str],
            embed_matrix: Optional[np.ndarray]=None,
    ) -> None:
        self._token2index = token2index
        self._index2token = index2token
        self._vocab_size = len(self._token2index)
        self._embed_matrix = embed_matrix

    @property
    def embed_matrix(self) -> np.ndarray:
        '''
        get embed_matrix

        :return: pre-trained embed_matrix
        '''
        return self._embed_matrix

    @property
    def vocab_size(self) -> int:
        '''
        get vocabulary size.

        :return: vocabulary size.
        '''
        return self._vocab_size

    @classmethod
    def create(  # type: ignore
            cls,
            text_list: Sequence[str],
            max_vocab_size: Optional[int]=None,
            symbol_order: Optional[str]=None,
            train_embed_matrix: bool=False,
            dim_size: int=DEFAULT_EMBED_DIM,
            window_size: int=DEFAULT_W2V_WINDOW_SIZE,
    ) -> 'TextPreprocessor':
        '''
        Create vocabulary dict.
        all text has to be pre-tokenized.

        :param text_list: list of sentences.
        :param max_vocab_size: maximum vocabulary size.
        :param symbol_order: the order of special symbols.
        :param train_embed_matrix: train embed matrix or not. it consumes much time.
        :param dim_size: size of dimention for word embedding.
        :param window_size: window size for wird2vec.
        :return: TextPreprocessor
        '''
        assert isinstance(text_list, list)
        assert len(text_list) == 0 or isinstance(text_list[0], str)

        counter: collections.Counter = collections.Counter()
        for text in text_list:
            token_list = text.strip().split()
            counter.update(token_list)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        token_list, _ = zip(*count_pairs)

        token_list = cls.SYMBOLS + list(token_list)
        if max_vocab_size is not None:
            token_list = token_list[:max_vocab_size]

        token2index = dict(zip(token_list, range(len(token_list))))
        index2token = dict({index: token for token, index in token2index.items()})

        embed_matrix = None
        if train_embed_matrix:
            embed_matrix = cls.create_embed_matrix(
                text_list=text_list,
                index2token=index2token,
                dim=dim_size,
                window_size=window_size,
            )

        return cls(token2index, index2token, embed_matrix)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TextPreprocessor':
        '''
        load saved preprocessor.

        :param path: path to saved pickle file
        :return: TextPreprocessor
        '''
        if isinstance(path, str):
            path = Path(path)
        assert path.exists()

        with path.open('rb') as f:
            token2index, index2token, embed_matrix = pickle.loads(f.read())

        return cls(token2index, index2token, embed_matrix)

    def save(self, path: Union[str, Path]) -> None:
        '''
        save vocaburaly files.

        :param path: path to save
        '''
        if isinstance(path, str):
            path = Path(path)

        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with path.open('wb') as f:
            pickle.dump((self._token2index, self._index2token, self._embed_matrix), f)

    def token2index(self, token: str) -> int:
        '''
        convert token to index.

        :param token: token of sentence to be converted
        :return: index of token
        '''
        return self._token2index[token] if token in self._token2index else self.UNK_ID

    def index2token(self, index: int) -> str:
        '''
        convert index to token.

        :param index: index to be converted to token
        :retrun: token of sentence
        '''
        return self._index2token[index] if index in self._index2token else self.UNK_SYMBOL

    def tokens2indice(self, tokens: List[str], sos: bool=False, eos: bool=False) -> List[int]:
        '''
        convert list of tokens to list of index.

        :param tokens: list of token
        :return: list of index
        '''
        indices = list(self.token2index(token) for token in tokens)
        if sos:
            indices = [self.SOS_ID] + indices
        if eos:
            indices = [self.EOS_ID] + indices

        return indices

    def indice2tokens(self, indice: List[int], sos: bool=False, eos: bool=False) -> List[str]:
        '''
        convert list of index to list of tokens.

        :param indice: list of lidex
        :return: list of tokens
        '''
        tokens = list(self.index2token(index) for index in indice)
        if sos:
            tokens = [self.SOS_SYMBOL] + tokens
        if eos:
            tokens = [self.EOS_SYMBOL] + tokens

        return tokens

    @classmethod
    def create_embed_matrix(
            cls,
            text_list: Sequence[str],
            index2token: Mapping[int, str],
            dim: int=DEFAULT_EMBED_DIM,
            window_size: int=DEFAULT_W2V_WINDOW_SIZE,
            seed: int=4545,
    ) -> np.ndarray:
        '''
        create and train embed_matrix

        :param text_list: list of sentences.
        :param index2token: dict of index2token.
        :param dim: dimension of embedding.
        :param window_size: window size for word2vec.
        :return: embed_matrix.
        '''
        word_list_list = [text.strip().split() for text in text_list]
        model = Word2Vec(
            sentences=word_list_list,
            size=dim,
            window=window_size,
            min_count=0,
            sample=0.0,
        )

        embed_matrix = []
        np.random.seed(seed)
        for i in range(len(index2token)):
            token = index2token[i]
            if token in model.wv.vocab:
                embed_matrix.append(model.wv[token])
            else:
                embed_matrix.append(np.random.standard_normal(dim).astype(np.float32))

        return np.array(embed_matrix)

    def fix_symbol_order(self, symbol_list: List[str]) -> None:
        '''
        change the order of symbols.
        structure of symbol_list is like ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

        :param symbol_list: symbol list
        '''
        self.PAD_ID = symbol_list.index(self.PAD_SYMBOL)
        self.SOS_ID = symbol_list.index(self.SOS_SYMBOL)
        self.EOS_ID = symbol_list.index(self.EOS_SYMBOL)
        self.UNK_ID = symbol_list.index(self.UNK_SYMBOL)
        self.SYMBOLS = symbol_list


if __name__ == '__main__':
    parser = ArgumentParser('create vocaburaly with Flickr8k dataset')
    parser.add_argument('-normal_path', type=str,
                        help='path to normal captions, which are pre-processed')
    parser.add_argument('-style_path', type=str,
                        help='path to style captions, which are pre-processed')
    parser.add_argument('-save_path', type=str,
                        help='path to save path')
    parser.add_argument('-max_vocab_size', type=int, default=0,
                        help='maximum vocaburaly size. all vocaburaly is used when 0')
    parser.add_argument('-train_embed_matrix', action='store_true',
                        help='train embed matrix')
    parser.add_argument('-dim_size', type=int, default=512,
                        help='dimension of word embedding')
    parser.add_argument('-window_size', type=int, default=5,
                        help='window size used for word2vec')
    args = parser.parse_args()

    normal_path = Path(args.normal_path)
    style_path = Path(args.style_path)
    save_path = Path(args.save_path)
    assert normal_path.exists()
    assert style_path.exists()

    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)

    deliminator = re.compile(r'\d*.jpg#\d*')

    with normal_path.open() as f:
        normal_data = [sentence.strip() for sentence in f.readlines()]
        for idx, img_caption in enumerate(normal_data):
            caption = deliminator.sub('', img_caption)
            normal_data[idx] = caption.split()

    with style_path.open() as f:
        style_data = [sentence.strip() for sentence in f.readlines()]

    mixed_data = normal_data + style_data
    max_vocab_size = args.max_vocab_size if args.max_vocab_size > 0 else None

    print('Creating vocaburaly...')
    text_preprocessor = TextPreprocessor.create(
        text_list=mixed_data,
        max_vocab_size=max_vocab_size,
        train_embed_matrix=args.train_embed_matrix,
        dim_size=args.dim_size,
        window_size=args.window_size,
    )

    print(f'Saving vocaburaly files to {str(save_path)}')
    text_preprocessor.save(save_path)

    print('Done')
