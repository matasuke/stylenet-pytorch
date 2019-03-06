import re
from typing import Union, Optional, List, Tuple
from pathlib import Path

from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from text_preprocessor import TextPreprocessor


class ImageCaptionDataset(Dataset):
    '''
    Dataset for Image Captioning
    '''
    __slots__ = ['img_dir', 'imgid_caption_pair', 'text_preprocessor', 'transform']

    def __init__(
            self,
            img_dir: Union[str, Path],
            imgid_caption_pair: List[List[str, str]],
            text_preprocessor: TextPreprocessor,
            transform: Optional[transforms.Compose]=None,
    ) -> None:
        '''
        :param img_dir: path to image directory
        :param imgid_caption_pair: list of list, which contains img_id and caption
        :param text_preprocessor: TextPreprocessor
        :param transform: transform to be applied
        '''
        if isinstance(img_dir, str):
            img_dir = Path(img_dir)
        assert img_dir.exists()

        self.img_dir = img_dir
        self.imgid_caption_pair = imgid_caption_pair
        self.text_preprocessor = text_preprocessor
        self.transform = transform

    @classmethod
    def create(
            cls,
            img_dir: Union[str, Path],
            caption_path: Union[str, Path],
            preprocessor_path: Union[str, Path],
            transform: Optional[transforms.Compose]=None,
    ) -> 'ImageCaptionDataset':
        '''
        Create dataset for FlickrDataset

        :param img_dir: path to image directory
        :param caption_path: path to caption
        :param text_preprocessor: TextPreprocessor
        :param transform: transform to be applied
        '''
        if isinstance(img_dir, str):
            img_dir = Path(img_dir)
        if isinstance(caption_path, str):
            caption_path = Path(caption_path)
        if isinstance(preprocessor_path, str):
            preprocessor_path = Path(preprocessor_path)
        assert img_dir.exists()
        assert caption_path.exists()
        assert preprocessor_path.exists()

        with caption_path.open() as f:
            img_caption_list = [caption.strip().lower() for caption in f.readlines()]

        imgid_caption_pair = []
        diliminator = re.compile(r'#\d*')
        for img_caption in img_caption_list:
            imgid_caption = diliminator.split(img_caption)
            imgid_caption = [i.strip() for i in img_caption]
            imgid_caption_pair.append(imgid_caption)

        text_preprocessor = TextPreprocessor.load(preprocessor_path)

        return cls(img_dir, imgid_caption_pair, text_preprocessor, transform)

    def __len__(self):
        return len(self.caption_list)

    def __getitems__(self, idx: int):
        img_name = self.img_names[idx]
        img_name = self.img_dir / img_name
        caption = self.caption_list[idx]

        image = io.imread(img_name.as_posix())
        if self.transform:
            image = self.transform(image)

        # convert caption to indice
        tokens = caption.split()
        indice = self.text_preprocessor.tokens2indice(tokens, sos=True, eos=True)

        return image, indice

class CaptionDataset(Dataset):
    '''
    Dataset for only captions
    Dataset has to be pre-tokenized
    '''
    __slots__ = ['caption_list', 'text_preprocessor']

    def __init__(
            self,
            caption_list: List[str],
            text_preprocessor: 'TextPreprocessor',
    ):
        '''
        create style caption dataset.

        :param caption_list: list of captions
        :param text_preprocessor: text preprocessor
        '''
        self.caption_list = caption_list
        self.text_preprocessor = text_preprocessor

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, idx: int) -> str:
        tokens = self.caption_list[idx].split()
        indices = self.text_preprocessor.tokens2indice(tokens, eos=True)
        indices = torch.Tensor(indices)

        return indices

    @classmethod
    def create(
            cls,
            caption_path: Union[str, Path],
            preprocessor_path: Union[str, Path],
    ) -> 'CaptionDataset':
        '''
        Create dataset for only captions

        :param caption_path: path to caption file.
        '''
        if isinstance(caption_path, str):
            caption_path = Path(caption_path)
        if isinstance(preprocessor_path, str):
            preprocessor_path = Path(preprocessor_path)
        assert caption_path.exists()
        assert preprocessor_path.exists()

        with caption_path.open() as f:
            caption_list = [caption.strip().lower() for caption in f.readlines()]
        text_preprocessor = TextPreprocessor.load(preprocessor_path)

        cls(caption_list, text_preprocessor)

class Rescale:
    '''
    rescale the image in a sample to a given size.
    '''
    def __init__(self, output_size: Union[int, Tuple[int, int]]) -> None:
        '''
        :param output_size: desired output size.
        '''
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image: 'np.ndarray'):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))

        return img

def get_image_caption_loader(
        img_dir: Union[str, Path],
        caption_path: Union[str, Path],
        preprocessor_path: Union[str, Path],
        batch_size: int=128,
        transform: Optional[transforms.Compose]=None,
        shuffle: bool=True,
        num_workers: int=0,
        img_size: Tuple[int, int]=(224, 224),
):
    '''
    get image caption dataloader with preprocessing

    :param img_dir: image directory
    :param caption_path: path to caption file
    :param preprocessor: path to text preprocessor
    :param batch_size: batch size
    :param transform: pre-processing method
    :param shuffle: shuffle data on every epoch
    :param num_workers: the number of workers
    '''
    if isinstance(img_dir, str):
        img_dir = Path(img_dir)
    if isinstance(caption_path, str):
        caption_path = Path(caption_path)
    if isinstance(preprocessor_path, str):
        preprocessor_path = Path(preprocessor_path)
    assert img_dir.exists()
    assert caption_path.exists()
    assert preprocessor_path.exists()

    if transform is None:
        transform = transforms.Compose([
            Rescale(img_size),
            transforms.ToTensor(),
        ])

    image_caption_dataset = ImageCaptionDataset.create(
        img_dir=img_dir,
        caption_path=caption_path,
        preprocessor_path=preprocessor_path,
        transform=transform,
    )

    data_loader = DataLoader(
        dataset=image_caption_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=img_caption_collate_fn,
    )

    return data_loader

def get_caption_data_loader(
        caption_path: Union[str, Path],
        preprocessor_path: Union[str, Path],
        batch_size: int=128,
        shuffle: bool=True,
        num_workers: int=0,
) -> DataLoader:
    '''
    get data loader for stylized captions.

    :param caption_path: path to caption file
    :param preprocessor: path to text preprocessor
    :param batch_size: batch size
    :param shuffle: shuffle data on every epoch
    :param num_workers: the number of workers
    '''
    if isinstance(caption_path, str):
        caption_path = Path(caption_path)
    if isinstance(preprocessor_path, str):
        preprocessor_path = Path(preprocessor_path)
    assert caption_path.exists()
    assert preprocessor_path.exists()

    caption_dataset = CaptionDataset.create(
        caption_path=caption_path,
        preprocessor_path=preprocessor_path,
    )

    data_loader = DataLoader(
        dataset=caption_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=caption_collate_fn,
    )

    return data_loader

def merge(sequences: List[torch.Tensor]):
    lengths = torch.LongTensor([len(seq) for seq in sequences])
    padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq[:end]
    return padded_seqs, lengths

def img_caption_collate_fn(
        img_caption: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, ...]:
    '''
    create mini-batch tensors from data (image, caption)
    use this collate_fn to pad captions.

    :param data: mini batch
    '''
    # sort a list of caption length to use pack_padded_sequence
    img_caption.sort(key=lambda x: len(x[1]), reverse=True)

    # unpack image and sequences
    images, captions = zip(*img_caption)

    # convert tuple of 3D tensor to 4D tensor
    images = torch.stack(images, dim=0)

    # convert tuple of 1D tensor to 2D tensor
    captions, lengths = merge(captions)

    return images, captions, lengths

def caption_collate_fn(captions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    create mini-batch tensors from caption

    :param caption: mini batch
    '''
    # sort a list of caption length to use pack_padded_sequence
    captions.sort(key=lambda x: len(x[1]), reverse=True)

    # convert tuple of 1D tensor to 2D tensor
    captions, lengths = merge(captions)

    return captions, lengths
