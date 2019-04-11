from argparse import ArgumentParser
from pathlib import Path
from typing import Union, Optional, Tuple

from PIL import Image
import torch
from torchvision import transforms

from modules import EncoderCNN, FactoredLSTM
from config_loader import HParams
from text_preprocessor import TextPreprocessor


def load_images(
        img_dir: Union[str, Path],
        transform: Optional[transforms.Compose]=None
) -> Tuple:
    '''
    load image from specified directory
    '''
    PREFIX = '*.jpg'

    if isinstance(img_dir, str):
        img_dir = Path(img_dir)
    assert img_dir.exists()

    name_list = []
    img_list = []
    for img_name in img_dir.glob(PREFIX):

        img = Image.open(img_name.as_posix())
        if transform is not None:
            img = transform(img).unsqueeze(0)

        name_list.append(img_name.as_posix())
        img_list.append(img)

    return name_list, img_list

def main(hparams: HParams):
    '''
    generate captions from images
    '''
    device = torch.device(hparams.gpus if torch.cuda.is_available() else 'cpu')
    text_preprocessor = TextPreprocessor.load(hparams.text_preprocessor_path)

    transform = transforms.Compose([
        transforms.Resize([hparams.crop_size, hparams.crop_size]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # build model
    encoder = EncoderCNN(hparams.hidden_dim).eval()
    decoder = FactoredLSTM(hparams.embed_dim, text_preprocessor.vocab_size, hparams.hidden_dim,
                           hparams.style_dim, hparams.num_layers, train=False, device=device)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    checkpoints = torch.load(hparams.checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoints['encoder'])
    decoder.load_state_dict(checkpoints['decoder'])

    img_names, img_list = load_images(hparams.img_dir, transform)
    for idx, (img_name, img) in enumerate(zip(img_names, img_list)):
        img = img.to(device)
        features = encoder(img)

        if hparams.decoder == 'greedy':
            output = decoder.sample_greedy(features, hparams.gen_max_len, hparams.mode,
                                           text_preprocessor.SOS_ID, text_preprocessor.EOS_ID)
            output = output[0].cpu().tolist()
        else:
            output = decoder.sample_beam(features, hparams.beam_width, hparams.gen_max_len, hparams.mode,
                                         text_preprocessor.SOS_ID, text_preprocessor.EOS_ID)

        output = output[1:output.index(text_preprocessor.EOS_ID)]  # delete SOS and EOS
        caption = text_preprocessor.indice2tokens(output)

        print(img_names[idx])
        print(' '.join(token for token in caption))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-decoder', type=str, choices=['greedy', 'beam'], default='greedy',
                        help='Use CUDA on the listed devices.')
    parser.add_argument('-img_dir', type=str, required=True,
                        help='path to input image directory')
    parser.add_argument('-checkpoint_path', type=str, required=True,
                        help='path to trained model')
    parser.add_argument('-text_preprocessor_path', type=str, required=True,
                        help='path to text preprocessor')
    parser.add_argument('-mode', type=str, choices=['default', 'style'], default='style',
                        help='generating mode')
    parser.add_argument('-beam_width', type=int, default=10,
                        help='beam width')
    parser.add_argument('-gen_max_len', type=int, default=30,
                        help='maximum number of tokens to be generated')
    parser.add_argument('-config_path', type=str, required=True,
                        help='path to config file')
    parser.add_argument('-gpus', type=int, help='Use CUDA on the listed devices.')
    args = parser.parse_args()

    hparams = HParams.load(args.config_path)
    hparams.parse_and_add(args)

    main(hparams)
