#!sh/zsh

python stylenet/text_preprocessor.py \
    -normal_path data/train_data/captions/Flickr8k.token.txt \
    -style_caption_path data/train_data/captions/FlickrStyle_v0.9/humor/funny_train.txt \
    -save_path data/trained_model/text_preprocessor/Flickr8k_FlickrStyle-humor.pkl \
