# Stylenet Pytorch implementation
Pytorch Implementation of [Stylenet](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/06/Generating-Attractive-Visual-Captions-with-Styles.pdf)

## Setup
```
pipenv install
pipenv shell
```

## Prepare dataset

Preprocess tokens for pytorch data loader.
```
python stylenet/text_preprocessor.py \
    -normal_path <path to non-style caption> \
    -style_path <path to style specific caption> \
    -save_path <path to save name of vocab file> \
    -max_vocab_size 0 # set 0 when using all vocaburaly. \
    # -train_embed_matrix # set this if you want to train pre-trained embedding. \
    -dim_size 512 \
    -window_size 5
```

or just put Flickr8K dataset and FlickrStyle to 'data/train_data' and type it.
'''
sh sh/preprocess.sh
'''

# Train
configuration for training and inference is based on data/configs/default.yml

'''
python stylenet/train.py -config_path data/configs/default.yml
'''

# inference
'''
python stylenet/generate.py \
    -img_dir <path to image directory>/ # it can be /sample_imgs/ \
    -text_preprocessor_path <path to save name of vocab file created by text_preprocessor.py> \
    -checkpoint_path <path to saved checkpoint> \
    -mode style # or default \
    -decoder beam # beam or greedy \
    -beam_width 5 \
    -gen_max_len 50 \
    -config_path data/configs/default.yml \
'''
