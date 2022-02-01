
#-*- coding: utf-8 -*-

import time

import os
import json
import math
import random
import argparse
import numpy as np
#from tqdm import tqdm
import pandas as pd

from token_list import libri_token

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim


from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau

import label_loader
from data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler

from models import EncoderRNN, DecoderRNN, Seq2Seq
from eval_distance import compute_cer, compute_wer, get_transcripts


char2index = dict()
index2char = dict()
SOS_token = 0
EOS_token = 0
PAD_token = 0

def test(train_name, model, data_loader, device, save_output=False):
    total_loss = 0.
    total_cer_dist, total_cer_length = 0, 0
    total_wer_dist, total_wer_length = 0, 0
    transcripts_list = []

    model.eval()
    with torch.no_grad():
        for i, (data) in enumerate(data_loader):

            # get data
            feats, scripts, feat_lengths, script_lengths = data  
            feats = feats.to(device)
            scripts = scripts.to(device)
            feat_lengths = feat_lengths.to(device)

            target = scripts[:, 1:] # remove SOS token

            # Forward
            logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=0)
            logit = torch.stack(logit, dim=1).to(device)
            y_hat = logit.max(-1)[1]   

            # CER Calculation
            cer_dist, cer_len = compute_cer(y_hat, target, EOS_token ,index2char)
            total_cer_dist += cer_dist
            total_cer_length +=cer_len

            # WER Calculation
            wer_dist, wer_len = compute_wer(y_hat, target, EOS_token ,index2char)     
            total_wer_dist += wer_dist
            total_wer_length +=wer_len

            # print sample 
            if save_output == True:
                transcripts = get_transcripts(y_hat, target, EOS_token ,index2char)
                ##################LOG######################################################
                with open('./log/'+train_name+"_TEST_Result.txt", "a") as f:
                    f.write(transcripts[0]+'\n')


    return float(total_cer_dist / total_cer_length) * 100 , float(total_wer_dist / total_wer_length) * 100



def main():
    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token

    parser = argparse.ArgumentParser(description='DSP_ASR')
    parser.add_argument('--model-name', type=str, default='DSP_ASR')

    ## Train Name for log 
    parser.add_argument('--train_name', type=str, default= 'TEST')
    # Load Pretrained Mode
    parser.add_argument('--load-model', action='store_true', default=False, help='Load model')
    parser.add_argument('--model-path', default='./saved_models/Libri_360_200Epoch.pth', help='model to load') 
    
    # PATH 
    parser.add_argument('--test-file', nargs='*',           
                        help='data list about test dataset', default='./Libri_data/Libri_test_list_short.csv')
    parser.add_argument('--dataset-path', default='./Libri_data', help='Target dataset path')
    parser.add_argument('--save-folder', default='./saved_models', help='Location to save epoch models')

    # Hyperparameters
    parser.add_argument('--rnn-type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
    parser.add_argument('--encoder_layers', type=int, default=3, help='number of layers of model (default: 3)')
    parser.add_argument('--encoder_size', type=int, default=512, help='hidden size of model (default: 512)')
    parser.add_argument('--decoder_layers', type=int, default=2, help='number of pyramidal layers (default: 2)')
    parser.add_argument('--decoder_size', type=int, default=512, help='hidden size of model (default: 512)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate in training (default: 0.3)')
    parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True, help='Turn off bi-directional RNNs, introduces lookahead convolution')
    # GPU and batch setting
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size in training (default: 256)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers in dataset loader (default: 4)')  # CPU Usage
    # Data
    parser.add_argument('--max_len', type=int, default=80, help='Maximum characters of sentence (default: 80)')
    parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
    # Audio Config
    parser.add_argument('--sample-rate', default=16000, type=int, help='Sampling Rate')
    parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram')
    parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram')
    # etc.       
    parser.add_argument('--seed', type=int, default=123456, help='random seed (default: 123456)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    char2index, index2char = label_loader.load_label_tokenList(libri_token)
    #print(char2index)
    SOS_token = char2index['<sos>']
    EOS_token = char2index['<eos>']
    PAD_token = char2index['<pad>']

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    print('Device:', device)

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride)

    # Batch Size
    batch_size = args.batch_size

    print(">> Test dataset at: ", args.test_file) 
    testData_list = pd.read_csv(args.test_file)

    test_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                    dataset_path=args.dataset_path, 
                                    data_list=testData_list,
                                    char2index=char2index, sos_id=SOS_token, eos_id=EOS_token,
                                    normalize=True,
                                    SAflag = False)
    testLoader = AudioDataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers)

    #print("Model Initialize..")
    input_size = int(math.floor((args.sample_rate * args.window_size) / 2) + 1)
    enc = EncoderRNN(input_size, args.encoder_size, n_layers=args.encoder_layers,
                     dropout_p=args.dropout, bidirectional=args.bidirectional, 
                     rnn_cell=args.rnn_type, variable_lengths=False)

    dec = DecoderRNN(len(char2index), args.max_len, args.decoder_size, args.encoder_size,
                     SOS_token, EOS_token,
                     n_layers=args.decoder_layers, rnn_cell=args.rnn_type, 
                     dropout_p=args.dropout, bidirectional_encoder=args.bidirectional)


    model = Seq2Seq(enc, dec)

    print("Loading checkpoint model %s" % args.model_path)
    state = torch.load(args.model_path)
    model.load_state_dict(state['model'])
    print('Model loaded')
    best_cer = state['best_cer'] 
    best_wer = state['best_wer']
    print("best WER ", best_wer)
    begin_epoch = state['Epochs'] + 1

    model = model.to(device)
    train_start = time.time()
    # Test
    test_cer, test_wer = test(args.train_name, model, testLoader, device, save_output=True)
    test_log = '\nAverage CER {cer:.3f}%\tAverage WER {wer:.3f}%'.format( cer=test_cer, wer=test_wer)
    print(test_log)

    # Time
    Train_time_sofar = time.time() - train_start
    Taken_hour = Train_time_sofar//3600
    Taken_min = (Train_time_sofar - Taken_hour*3600)//60
    Time_info = "{}(h) {}(m) Taken".format(Taken_hour, Taken_min )
    print(Time_info)

    # Log Summary
    with open('./log/'+args.train_name+"_TEST_Result.txt", "a") as ff:
        ff.write(test_log + "\n")
        ff.write(Time_info + "\n\n")

if __name__ == "__main__":
    main()
