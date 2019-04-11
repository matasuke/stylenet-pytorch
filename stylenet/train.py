from argparse import ArgumentParser
from pathlib import Path
import warnings
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

from config_loader import HParams
from modules import EncoderCNN, FactoredLSTM
from data_loader import get_image_caption_loader, get_caption_loader
from text_preprocessor import TextPreprocessor
from optim import Optim


def main(hparams: HParams):
    '''
    setup training.
    '''
    if torch.cuda.is_available() and not hparams.gpus:
        warnings.warn('WARNING: you have a CUDA device, so you should probably run with -gpus 0')

    device = torch.device(hparams.gpus if torch.cuda.is_available() else 'cpu')

    # data setup
    print(f"Loading vocabulary...")
    text_preprocessor = TextPreprocessor.load(hparams.preprocessor_path)

    transform = transforms.Compose([
        transforms.Resize([hparams.img_size, hparams.img_size]),
        transforms.RandomCrop([hparams.crop_size, hparams.crop_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # create dataloader
    print('Creating DataLoader...')
    normal_data_loader = get_image_caption_loader(
        hparams.img_dir,
        hparams.normal_caption_path,
        text_preprocessor,
        hparams.normal_batch_size,
        transform,
        shuffle=True,
        num_workers=hparams.num_workers,
    )

    style_data_loader = get_caption_loader(
        hparams.style_caption_path,
        text_preprocessor,
        batch_size=hparams.style_batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
    )

    if hparams.train_from:
        # loading checkpoint
        print('Loading checkpoint...')
        checkpoint = torch.load(hparams.train_from)
    else:
        normal_opt = Optim(
            hparams.optimizer,
            hparams.normal_lr,
            hparams.max_grad_norm,
            hparams.lr_decay,
            hparams.start_decay_at,
        )
        style_opt = Optim(
            hparams.optimizer,
            hparams.style_lr,
            hparams.max_grad_norm,
            hparams.lr_decay,
            hparams.start_decay_at,
        )

    print('Building model...')
    encoder = EncoderCNN(hparams.hidden_dim)
    decoder = FactoredLSTM(hparams.embed_dim, text_preprocessor.vocab_size, hparams.hidden_dim,
                           hparams.style_dim, hparams.num_layers, hparams.random_init,
                           hparams.dropout_ratio, train=True, device=device)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=text_preprocessor.PAD_ID)
    normal_params = list(encoder.parameters()) + list(decoder.default_parameters())
    style_params = list(decoder.style_parameters())
    normal_opt.set_parameters(normal_params)
    style_opt.set_parameters(style_params)

    if hparams.train_from:
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        normal_opt.load_state_dict(checkpoint['normal_opt'])
        style_opt.load_state_dict(checkpoint['style_opt'])

    # traininig loop
    print('Start training...')
    for epoch in range(hparams.num_epoch):

        # result
        sum_normal_loss, sum_style_loss, sum_normal_ppl, sum_style_ppl = 0, 0, 0, 0

        # normal caption
        for i, (images, in_captions, out_captions, lengths) in enumerate(normal_data_loader):
            images = images.to(device)
            in_captions = in_captions.to(device)
            out_captions = out_captions.contiguous().view(-1).to(device)

            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(in_captions, features, mode='default')
            loss = criterion(outputs.view(-1, outputs.size(-1)), out_captions)
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            normal_opt.step()

            # print log
            sum_normal_loss += loss.item()
            sum_normal_ppl += np.exp(loss.item())
            if i % hparams.normal_log_step == 0:
                print(f'Epoch [{epoch}/{hparams.num_epoch}], Normal Step: [{i}/{len(normal_data_loader)}] '
                      f'Normal Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):5.4f}')

        # style caption
        for i, (in_captions, out_captions, lengths) in enumerate(style_data_loader):
            in_captions = in_captions.to(device)
            out_captions = out_captions.contiguous().view(-1).to(device)

            # Forward, backward and optimize
            outputs = decoder(in_captions, None, mode='style')
            loss = criterion(outputs.view(-1, outputs.size(-1)), out_captions)

            decoder.zero_grad()
            loss.backward()
            style_opt.step()

            sum_style_loss += loss.item()
            sum_style_ppl += np.exp(loss.item())
            # print log
            if i % hparams.style_log_step == 0:
                print(f'Epoch [{epoch}/{hparams.num_epoch}], Style Step: [{i}/{len(style_data_loader)}] '
                      f'Style Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):5.4f}')

        model_params = {
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'epoch': epoch,
            'normal_opt': normal_opt.optimizer.state_dict(),
            'style_opt': style_opt.optimizer.state_dict(),
        }

        avg_normal_loss = sum_normal_loss / len(normal_data_loader)
        avg_style_loss = sum_style_loss / len(style_data_loader)
        avg_normal_ppl = sum_normal_ppl / len(normal_data_loader)
        avg_style_ppl = sum_style_ppl / len(style_data_loader)
        print(f'Epoch [{epoch}/{hparams.num_epoch}] statistics')
        print(f'Normal Loss: {avg_normal_loss:.4f} Normal ppl: {avg_normal_ppl:5.4f} '
              f'Style Loss: {avg_style_loss:.4f} Style ppl: {avg_style_ppl:5.4f}')

        torch.save(
            model_params,
            f'{hparams.model_path}/n-loss_{avg_normal_loss:.4f}_s-loss_{avg_style_loss:.4f}_'
            f'n-ppl_{avg_normal_ppl:5.4f}_s-ppl_{avg_style_ppl:5.4f}_epoch_{epoch}.pt'
        )


if __name__ == '__main__':
    parser = ArgumentParser(description='Train stylenet')
    parser.add_argument('-config_path', type=str, default='',
                        help='path to YML format config file')
    parser.add_argument('-model_path', type=str,
                        help='path to save trained model')
    parser.add_argument('-train_from', type=str, default='',
                        help='checkpoint to load')
    parser.add_argument('-preprocessor_path', type=str,
                        help='text preprocessor')
    parser.add_argument('-img_dir', type=str,
                        help='path to image directory')
    parser.add_argument('-normal_caption_path', type=str,
                        help='path to normal caption path')
    parser.add_argument('-style_caption_path', type=str,
                        help='path to style caption path')
    parser.add_argument('-normal_batch_size', type=int, default=64,
                        help='normal batch size')
    parser.add_argument('-style_batch_size', type=int, default=96,
                        help='style batch size')
    parser.add_argument('-random_init', type=float, default=1.0,
                        help='paramteters for factoredLSTM are initialized over \
                        uniform distribution with support (-random_init, random_init)')
    parser.add_argument('-dropout_ratio', type=float, default=0.3,
                        help='dropout ratio')
    parser.add_argument('-num_layers', type=int, default=1,
                        help='the number of FactoredLSTM layer')
    parser.add_argument('-embed_dim', type=int, default=300,
                        help='dimention of embedding layer')
    parser.add_argument('-hidden_dim', type=int, default=512,
                        help='dimention of hidden layer')
    parser.add_argument('-style_dim', type=int, default=512,
                        help='dimention of style factor in factoredLSTM')
    parser.add_argument('-optimizer', type=str, default='adam',
                        choices=['sgd', 'adagrad', 'adadelta', 'adam'],
                        help='type of optimizer')
    parser.add_argument('-max_grad_norm', type=float, default=5.0,
                        help='maximum gradient norm')
    parser.add_argument('-lr_decay', type=int, default=1,
                        help='decay ratio for learning rate, does not decay when 1')
    parser.add_argument('-start_decay_at', type=int, default=1000,
                        help='decay learning rate at specified epoch')
    parser.add_argument('-normal_lr', type=float, default=0.0002,
                        help='learning rate for normal caption')
    parser.add_argument('-style_lr', type=float, default=0.0005,
                        help='learning rate for style caption')
    parser.add_argument('-num_epoch', type=int, default=30,
                        help='epoch to be executed')
    parser.add_argument('-img_size', type=int, default=256,
                        help='size for resizing images')
    parser.add_argument('-crop_size', type=int, default=224,
                        help='size for randomly cropping images')

    # logger
    parser.add_argument('-normal_log_step', type=int, default=50,
                        help='steps to show log of normal training')
    parser.add_argument('-style_log_step', type=int, default=10,
                        help='steps to show log of style training')

    # GPU
    parser.add_argument('-gpus', type=int, default=[], nargs='+',
                        help='Use CUDA on the listed devices.')
    parser.add_argument('-num_workers', type=int, default=0,
                        help='the number of workers')
    args = parser.parse_args()

    if args.config_path:
        config_path = Path(args.config_path)
        assert config_path.exists()
        hparams = HParams.load(config_path)
    else:
        hparams = HParams.parse(args)

    print(hparams)

    model_path = Path(hparams.model_path)
    if not model_path.exists():
        model_path.mkdir(parents=True)

    main(hparams)
