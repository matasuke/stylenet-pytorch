from pathlib import Path
import re

from PIL import Image
from torchvision import transforms

from stylenet.text_preprocessor import TextPreprocessor
from stylenet.data_loader import (
    ImageCaptionDataset,
    CaptionDataset,
    Rescale,
    get_image_caption_loader,
    get_caption_loader
)


TEXT_PREPROCESSOR_PATH = Path('test/data/test_text_preprocessor.pkl')
IMG_DIR_PATH = Path('test/data/images')
CAPTION_PATH = Path('test/data/captions.txt')
DEFAULT_IMG_SIZE = 256


def test_image_caption_dataset():
    '''
    Class for testing ImageCaptionDataset
    '''
    text_preprocessor = TextPreprocessor.load(TEXT_PREPROCESSOR_PATH)

    transform = transforms.Compose([
        transforms.Resize([DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE]),
        transforms.ToTensor(),
    ])

    image_caption_data = ImageCaptionDataset.create(
        img_dir=IMG_DIR_PATH,
        caption_path=CAPTION_PATH,
        text_preprocessor=text_preprocessor,
        transform=transform,
    )

    assert len(image_caption_data) == len(CAPTION_PATH.open().readlines())
