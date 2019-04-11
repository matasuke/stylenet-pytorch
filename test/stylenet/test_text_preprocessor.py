import numpy as np
from pathlib import Path

from stylenet.text_preprocessor import TextPreprocessor

EXPECT_SENTENCE = 'the weather today is very nice .'
EXPECT_TOKENS = ['the', 'weather', 'today', 'is', 'very', 'nice', '.']

class TestTextPreprocessor:
    """
    Class for testing TextPreprocessor
    """

    def test_create(self):
        text_list = [EXPECT_SENTENCE]
        expect_dim_size = 10
        expect_window_size = 2
        preprocessor = TextPreprocessor.create(
            text_list,
            train_embed_matrix=True,
            dim_size=expect_dim_size,
            window_size=expect_window_size,
        )
        expect_vocab_size = len(preprocessor.SYMBOLS) + len(EXPECT_TOKENS)

        assert len(preprocessor._token2index) == expect_vocab_size
        assert len(preprocessor._index2token) == expect_vocab_size
        assert preprocessor._embed_matrix.shape == (expect_vocab_size, expect_dim_size)

    def test_symbols_id(self):
        text_list = [EXPECT_SENTENCE]
        preprocessor = TextPreprocessor.create(text_list)

        # Check whether or not the symbols are assigned to specified ID.
        assert preprocessor.token2index('<PAD>') == 0
        assert preprocessor.token2index('<SOS>') == 1
        assert preprocessor.token2index('<EOS>') == 2
        assert preprocessor.token2index('<UNK>') == 3

        # Check the symbols can be found by reverse search.
        assert preprocessor.index2token(0) == '<PAD>'
        assert preprocessor.index2token(1) == '<SOS>'
        assert preprocessor.index2token(2) == '<EOS>'
        assert preprocessor.index2token(3) == '<UNK>'

    def test_token2index(self):
        text_list = [EXPECT_SENTENCE]
        preprocessor = TextPreprocessor.create(text_list)
        assert preprocessor.token2index('the') == 4
        assert preprocessor.token2index('bad') == TextPreprocessor.UNK_ID

    def test_index2token(self):
        text_list = [EXPECT_SENTENCE]
        preprocessor = TextPreprocessor.create(text_list)
        assert preprocessor.index2token(4) == 'the'

    def test_tokens2indice(self):
        text_list = [EXPECT_SENTENCE]
        preprocessor = TextPreprocessor.create(text_list)
        convert_tokens = ['the', 'weather', 'today']
        word_1 = preprocessor.token2index('the')
        word_2 = preprocessor.token2index('weather')
        word_3 = preprocessor.token2index('today')
        assert preprocessor.tokens2indice(convert_tokens) == [word_1, word_2, word_3]

    def test_indice2tokens(self):
        text_list = [EXPECT_SENTENCE]
        preprocessor = TextPreprocessor.create(text_list)
        converted = preprocessor.indice2tokens([4, 5, 6, 7, 8, 9, 10])
        assert converted == EXPECT_TOKENS

    def test_save_and_load(self, tmpdir):
        text_list = [EXPECT_SENTENCE]
        expect_dim_size = 5
        expect_window_size = 2
        preprocessor = TextPreprocessor.create(
            text_list,
            train_embed_matrix=True,
            dim_size=expect_dim_size,
            window_size=expect_window_size,
        )

        expect_save_path = Path(tmpdir.join('test_features.pkl'))
        preprocessor.save(expect_save_path)
        assert expect_save_path.exists()
        assert expect_save_path.is_file()

        loaded_preprocessor = TextPreprocessor.load(expect_save_path)

        expect_size = len(loaded_preprocessor.SYMBOLS) + len(EXPECT_TOKENS)
        assert len(loaded_preprocessor._token2index) == expect_size
        assert len(loaded_preprocessor._index2token) == expect_size
        assert len(loaded_preprocessor._embed_matrix[0]) == expect_dim_size

        # check whether or not specified symbol is assigned to designated ID.
        assert loaded_preprocessor.token2index('<PAD>') == 0
        assert loaded_preprocessor.token2index('<SOS>') == 1
        assert loaded_preprocessor.token2index('<EOS>') == 2
        assert loaded_preprocessor.token2index('<UNK>') == 3

        # Check the symbols can be found by reverse search.
        assert preprocessor.index2token(0) == '<PAD>'
        assert preprocessor.index2token(1) == '<SOS>'
        assert preprocessor.index2token(2) == '<EOS>'
        assert preprocessor.index2token(3) == '<UNK>'

    def test_array_type_is_nparray(self):
        text_list = [EXPECT_SENTENCE]
        expect_dim_size = 5
        expect_window_size = 2
        preprocessor = TextPreprocessor.create(
            text_list,
            train_embed_matrix=True,
            dim_size=expect_dim_size,
            window_size=expect_window_size,
        )
        assert isinstance(preprocessor.embed_matrix, np.ndarray)
        for row in preprocessor.embed_matrix:
            assert type(row[0]) == np.float32

    def test_fix_symbol_order(self):
        text_list = [EXPECT_SENTENCE]
        expect_symbol_order = ['<UNK>', '<EOS>', '<PAD>', '<SOS>']
        preprocessor = TextPreprocessor.create(text_list)
        preprocessor.fix_symbol_order(expect_symbol_order)

        assert preprocessor.UNK_ID == 0
        assert preprocessor.EOS_ID == 1
        assert preprocessor.PAD_ID == 2
        assert preprocessor.SOS_ID == 3
