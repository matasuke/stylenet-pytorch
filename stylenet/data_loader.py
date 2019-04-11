import re
from typing import Union, Optional, List, Tuple
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from stylenet.text_preprocessor import TextPreprocessor


DEFAULT_IMG_SIZE = (224, 224)


class ImageCaptionDataset(Dataset):
    '''
    Dataset for Image Captioning
    '''
    __slots__ = ['img_dir', 'imgid_caption_pairs', 'text_preprocessor', 'transform']

    def __init__(
            self,
            img_dir: Union[str, Path],
            img_caption_pairs: List[List[str]],
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
        self.img_caption_pairs = img_caption_pairs
        self.text_preprocessor = text_preprocessor
        self.transform = transform

    @classmethod
    def create(
            cls,
            img_dir: Union[str, Path],
            caption_path: Union[str, Path],
            text_preprocessor: TextPreprocessor,
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
        assert img_dir.exists()
        assert caption_path.exists()

        with caption_path.open() as f:
            img_caption_list = [caption.strip().lower() for caption in f.readlines()]

        img_caption_pairs = []
        diliminator = re.compile(r'#\d*\t')
        for img_caption in img_caption_list:
            img_name, caption = diliminator.split(img_caption)
            img_caption_pairs.append([img_name, caption])

        return cls(img_dir, img_caption_pairs, text_preprocessor, transform)

    def __len__(self):
        return len(self.img_caption_pairs)

    def __getitem__(self, idx: int):
        img_name, caption = self.img_caption_pairs[idx]
        img_path = self.img_dir / img_name
        image = Image.open(img_path.as_posix()).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # convert caption to indice
        tokens = caption.split()
        indice = self.text_preprocessor.tokens2indice(tokens, sos=True, eos=True)
        indice = torch.Tensor(indice)

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
            text_preprocessor: TextPreprocessor,
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

    def __getitem__(self, idx: int) -> torch.Tensor:
        tokens = self.caption_list[idx].split()
        indices = self.text_preprocessor.tokens2indice(tokens, sos=True, eos=True)
        indices = torch.Tensor(indices)

        return indices

    @classmethod
    def create(
            cls,
            caption_path: Union[str, Path],
            text_preprocessor: TextPreprocessor
    ) -> 'CaptionDataset':
        '''
        Create dataset for only captions

        :param caption_path: path to caption file.
        '''
        if isinstance(caption_path, str):
            caption_path = Path(caption_path)
        assert caption_path.exists()

        with caption_path.open() as f:
            caption_list = [caption.strip().lower() for caption in f.readlines()]

        return cls(caption_list, text_preprocessor)


def get_image_caption_loader(
        img_dir: Union[str, Path],
        caption_path: Union[str, Path],
        text_preprocessor: TextPreprocessor,
        batch_size: int=128,
        transform: Optional[transforms.Compose]=None,
        shuffle: bool=True,
        num_workers: int=0,
):
    '''
    get image caption dataloader with preprocessing

    :param img_dir: image directory
    :param caption_path: path to caption file
    :param text_preprocessor: TextPreprocessor
    :param batch_size: batch size
    :param transform: pre-processing method
    :param shuffle: shuffle data on every epoch
    :param num_workers: the number of workers
    '''
    if isinstance(img_dir, str):
        img_dir = Path(img_dir)
    if isinstance(caption_path, str):
        caption_path = Path(caption_path)
    assert img_dir.exists()
    assert caption_path.exists()

    if transform is None:
        transform = transforms.Compose([
            Rescale(DEFAULT_IMG_SIZE),
            transforms.ToTensor(),
        ])

    image_caption_dataset = ImageCaptionDataset.create(
        img_dir=img_dir,
        caption_path=caption_path,
        text_preprocessor=text_preprocessor,
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

def get_caption_loader(
        caption_path: Union[str, Path],
        text_preprocessor: TextPreprocessor,
        batch_size: int=128,
        shuffle: bool=True,
        num_workers: int=0,
) -> DataLoader:
    '''
    get data loader for stylized captions.

    :param caption_path: path to caption file
    :param text_preprocessor: TextPreprocessor
    :param batch_size: batch size
    :param shuffle: shuffle data on every epoch
    :param num_workers: the number of workers
    '''
    if isinstance(caption_path, str):
        caption_path = Path(caption_path)
    assert caption_path.exists()

    caption_dataset = CaptionDataset.create(
        caption_path=caption_path,
        text_preprocessor=text_preprocessor,
    )

    data_loader = DataLoader(
        dataset=caption_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=caption_collate_fn,
    )

    return data_loader

def merge(captions: List[torch.Tensor]):
    '''
    pad sequences for source and target.
    '''

    lengths = [len(cap) for cap in captions]
    in_padded_seqs = torch.zeros(len(captions), max(lengths)).long()
    out_padded_seqs = torch.zeros(len(captions), max(lengths)).long()

    for idx, cap in enumerate(captions):
        end = lengths[idx]
        in_padded_seqs[idx, :end-1] = cap[:end-1]
        out_padded_seqs[idx, :end-1] = cap[1:end]

    return in_padded_seqs, out_padded_seqs, lengths

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
    in_captions, out_captions, lengths = merge(captions)

    return images, in_captions, out_captions, lengths

def caption_collate_fn(captions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    '''
    create mini-batch tensors from caption

    :param caption: mini batch
    '''
    # sort a list of caption length to use pack_padded_sequence
    captions.sort(key=lambda x: len(x), reverse=True)

    # convert tuple of 1D tensor to 2D tensor
    in_captions, out_captions, lengths = merge(captions)

    return in_captions, out_captions, lengths
