from __future__ import print_function
import torch
from torchvision import datasets, models, transforms
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax
import numpy as np
import utils
from data_loader import get_coco_data_loader, get_basic_loader
from models import CNN, RNN
from vocab import Vocabulary, load_vocab
import os
from tqdm import tqdm

def beam_sample(decoder, features, max_len=25, k=5):
    batch_size, hidden_size = features.shape
    output_ids = torch.zeros(batch_size, k, max_len, dtype=torch.int).cuda()
    output_probs = torch.ones(batch_size, k, dtype=torch.int).cuda()
    hidden_state = torch.zeros(1, batch_size, k, hidden_size).cuda()
    cell_state = torch.zeros(1, batch_size, k, hidden_size).cuda()
    inputs = [features.unsqueeze(1) for i in range(k)] # (B, 1, H)
    vocab_size = decoder.linear.out_features

    for i in range(max_len):
        # pass data through recurrent network
        outputs = []
        for j in range(k):
            if i == 0:
                hiddens, states = decoder.unit(inputs[j], None)
            else:
                hiddens, states = decoder.unit(inputs[j], (hidden_state[:, :, j, :].contiguous(), cell_state[:, :, j, :].contiguous()))
            
            hidden_state[:, :, j, :], cell_state[:, :, j, :] = states

            logits = decoder.linear(hiddens.squeeze(1))
            outputs.append(log_softmax(logits, dim=1))
            outputs[j] += output_probs[:, j].unsqueeze(1).expand(batch_size, vocab_size)

        if i == 0:
            outputs = outputs[0]
        else:
            outputs = torch.cat(outputs, dim=1)

        top_k = outputs.topk(k, dim=1)
        output_probs, predicted = top_k
        branch_ids = predicted // vocab_size
        predicted %= vocab_size

        new_ids = torch.zeros(batch_size, k, max_len, dtype=torch.int)
        new_hidden_state = torch.zeros(*hidden_state.shape).cuda()
        new_cell_state   = torch.zeros(*cell_state.shape).cuda()
        for b in range(batch_size):
            new_ids[b] = output_ids[b, branch_ids[b]]
            new_ids[b, :, i] = predicted[b]

            new_hidden_state[:, b, :, :] = hidden_state[:, b, branch_ids[b], :]
            new_cell_state[:, b, :, :] = cell_state[:, b, branch_ids[b], :]
        
        output_ids = new_ids
        hidden_state = new_hidden_state
        cell_state = new_cell_state

        # prepare chosen words for next decoding step
        for j in range(k):
            inputs[j] = decoder.embeddings(predicted[:, j])
            inputs[j] = inputs[j].unsqueeze(1)
        # print("inputs shape: {}".format(inputs.shape))
    output_ids = output_ids[:, 0, :]
    print(output_ids.shape)
    return output_ids.squeeze()

def main(args):
    # hyperparameters
    batch_size = args.batch_size
    num_workers = 2

    # Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    vocab = load_vocab()

    loader = get_basic_loader(dir_path=os.path.join(args.image_path),
                              transform=transform,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)

    # Build the models
    embed_size = args.embed_size
    num_hiddens = args.num_hidden
    checkpoint_path = 'checkpoints'

    encoder = CNN(embed_size)
    decoder = RNN(embed_size,
                  num_hiddens,
                  len(vocab),
                  1,
                  rec_unit=args.rec_unit)

    encoder_state_dict, decoder_state_dict, optimizer, *meta = utils.load_models(
        args.checkpoint_file)
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Train the Models
    try:
        results = []
        with torch.no_grad():
            for step, (images, image_ids) in enumerate(tqdm(loader)):
                images = utils.to_var(images)

                features = encoder(images)
                captions = beam_sample(decoder, features)
                # captions = decoder.sample(features)
                captions = captions.cpu().data.numpy()
                captions = [
                    utils.convert_back_to_text(cap, vocab) for cap in captions
                ]
                captions_formatted = [{
                    'image_id': int(img_id),
                    'caption': cap
                } for img_id, cap in zip(image_ids, captions)]
                results.extend(captions_formatted)
                print('Sample:', captions_formatted[0])
    except KeyboardInterrupt:
        print('Ok bye!')
    finally:
        import json
        file_name = 'captions_model.json'
        with open(file_name, 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file',
                        type=str,
                        default=None,
                        help='path to saved checkpoint')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='size of batches')
    parser.add_argument('--rec_unit',
                        type=str,
                        default='gru',
                        help='choose "gru", "lstm" or "elman"')
    parser.add_argument('--image_path',
                        type=str,
                        default='data/test2014',
                        help='path to the directory of images')
    parser.add_argument('--embed_size',
                        type=int,
                        default='512',
                        help='number of embeddings')
    parser.add_argument('--num_hidden',
                        type=int,
                        default='512',
                        help='number of embeddings')
    args = parser.parse_args()
    main(args)
