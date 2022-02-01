
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

def train( train_name, model, data_loader, criterion, optimizer, scheduler, device, epoch, train_sampler, max_norm=400, teacher_forcing_ratio=1):
    total_loss = 0.
    total_cer_dist, total_cer_length = 0, 0
    total_wer_dist, total_wer_length = 0, 0

    model.train()
    for i, (data) in enumerate(data_loader):
        optimizer.zero_grad()

        # get Data 
        feats, scripts, feat_lengths, script_lengths = data

        feats = feats.to(device)
        scripts = scripts.to(device)
        feat_lengths = feat_lengths.to(device)

        # Forward
        logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=teacher_forcing_ratio)
        logit = torch.stack(logit, dim=1).to(device)
        y_hat = logit.max(-1)[1]

        # Loss
        target = scripts[:, 1:]     # remove sos token
        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
        total_loss += loss.item()

        # Backward 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()

        # CER Calculation
        cer_dist, cer_len = compute_cer(y_hat, target, EOS_token ,index2char)
        avg_cer = float(cer_dist/cer_len) * 100       
        total_cer_dist += cer_dist
        total_cer_length +=cer_len


        # WER Calculation
        wer_dist, wer_len = compute_wer(y_hat, target, EOS_token ,index2char)
        avg_wer = float(wer_dist/wer_len) * 100       
        total_wer_dist += wer_dist
        total_wer_length +=wer_len

        ##################LOG######################################################
        with open('./log/'+train_name+"_TrainLog.txt", "a") as f:
            f.write('Epoch %d batch %d/%d Loss %0.9f CER %0.4f WER %0.4f \n' %((epoch + 1), (i + 1), len(train_sampler), loss, avg_cer, avg_wer ))

    return total_loss / len(data_loader), float(total_cer_dist / total_cer_length) * 100 , float(total_wer_dist / total_wer_length) * 100


def evaluate(train_name, model, data_loader, criterion, device, save_output=False):
    total_loss = 0.
    total_cer_dist, total_cer_length = 0, 0
    total_wer_dist, total_wer_length = 0, 0
    total_sent_num = 0
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

            # Loss Calculation
            logit = logit[:,:target.size(1),:] 
            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()

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
                with open('./log/'+train_name+"_InfLog.txt", "a") as f:
                    f.write(transcripts[0]+'\n')

            total_sent_num += target.size(0)

    return total_loss / len(data_loader), float(total_cer_dist / total_cer_length) * 100 , float(total_wer_dist / total_wer_length) * 100



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
    
    # PATH 
    parser.add_argument('--train-file', type=str,
                        help='data list about train dataset', default='./Libri_data/Libri_train_360_list_short.csv') 
    parser.add_argument('--validation-file', nargs='*',           
                        help='data list about validation dataset', default='./Libri_data/Libri_val_list_short.csv') # = validation
    parser.add_argument('--dataset-path', default='./Libri_data', help='Target dataset path') 
    parser.add_argument('--save-folder', default='./saved_models', help='Location to save epoch models')

    # Load Pretrained Mode
    parser.add_argument('--model-path', default='./saved_models/Libri_360_200Epoch.pth', help='model to load') 
    parser.add_argument('--load-model', action='store_true', default=False, help='Load model')
    parser.add_argument('--finetune', dest='finetune', action='store_true', default=False,
                        help='Finetune the model after load model')
    # Hyperparameters
    parser.add_argument('--rnn-type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
    parser.add_argument('--encoder_layers', type=int, default=3, help='number of layers of model (default: 3)')
    parser.add_argument('--encoder_size', type=int, default=512, help='hidden size of model (default: 512)')
    parser.add_argument('--decoder_layers', type=int, default=2, help='number of pyramidal layers (default: 2)')
    parser.add_argument('--decoder_size', type=int, default=512, help='hidden size of model (default: 512)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate in training (default: 0.3)')
    parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True, help='Turn off bi-directional RNNs, introduces lookahead convolution')
    # GPU and batch setting
    parser.add_argument('--device_ids', nargs='*', default= [0,1,2,3,4])  # GPU id to use
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size in training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers in dataset loader (default: 4)')  # CPU Usage
    parser.add_argument('--num_gpu', type=int, default=5, help='Number of gpus (default: 1)')
    # learning setting
    parser.add_argument('--epochs', type=int, default=1, help='Number of max epochs in training (default: 200)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    parser.add_argument('--learning-anneal', default=1, type=float, help='Annealing learning rate every epoch')
    parser.add_argument('--teacher_forcing', type=float, default=0, help='Teacher forcing ratio in decoder (default: 1.0)')
    parser.add_argument('--max_len', type=int, default=80, help='Maximum characters of sentence (default: 80)')
    parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
    # Audio Config
    parser.add_argument('--sample-rate', default=16000, type=int, help='Sampling Rate')
    parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram')
    parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram')
    # etc.       
    parser.add_argument('--seed', type=int, default=123456, help='random seed (default: 123456)')
    parser.add_argument('--mode', type=str, default='train', help='Train or Test')

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
    args.num_gpu = torch.cuda.device_count() 


    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride)

    # Batch Size
    batch_size = args.batch_size

    print(">> Train dataset at: ", args.train_file) 
    print(">> Validation dataset at: ", args.validation_file) 
    trainData_list = pd.read_csv(args.train_file)
    validationData_list = pd.read_csv(args.validation_file) # = Validation

    if args.num_gpu != 1:
        # remove last some data which less than batch size
        last_batch = len(trainData_list) % batch_size
        if last_batch != 0 : 
            print("out ",last_batch, " batches")
            trainData_list = trainData_list[:-last_batch]
       
        last_batch = len(validationData_list) % batch_size
        if last_batch != 0 and last_batch < args.num_gpu:
            print("out ",last_batch, " batches")
            validationData_list = validationData_list[:-last_batch]


    #print("data Loading..")
    train_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                    dataset_path=args.dataset_path ,  
                                    data_list=trainData_list,
                                    char2index=char2index, sos_id=SOS_token, eos_id=EOS_token,
                                    normalize=True,
                                    SAflag = True) 

    train_sampler = BucketingSampler(train_dataset, batch_size=batch_size)
    train_loader = AudioDataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler)


    validation_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                    dataset_path=args.dataset_path, 
                                    data_list=validationData_list,
                                    char2index=char2index, sos_id=SOS_token, eos_id=EOS_token,
                                    normalize=True,
                                    SAflag = False)
    validationLoader = AudioDataLoader(validation_dataset, batch_size=batch_size, num_workers=args.num_workers)

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

    # check if a folder for saving trained model exists
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)

    best_cer = 1e10
    best_wer = 1e10
    optim_state = None
    begin_epoch = 0 

    if args.load_model:  # Starting from previous model
        print("Loading checkpoint model %s" % args.model_path)
        state = torch.load(args.model_path)
        model.load_state_dict(state['model'])
        print('Model loaded')
        best_cer = state['best_cer'] 
        best_wer = state['best_wer']
        print("best WER ", best_wer)
        begin_epoch = state['Epochs'] + 1

        if not args.finetune:  # Just load model
            optim_state = state['optimizer']
 
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)
    if optim_state is not None:
            optimizer.load_state_dict(optim_state)
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=PAD_token).to(device)

    #print(model)
    #print("Number of parameters: %d" % Seq2Seq.get_param_size(model))
    train_model = nn.DataParallel(model, device_ids = args.device_ids)

    train_start = time.time()

    for epoch in range(begin_epoch, args.epochs):

        # Train
        train_loss, train_cer, train_wer = train(args.train_name,train_model, train_loader, criterion, optimizer, scheduler , device, epoch, train_sampler, args.max_norm, args.teacher_forcing)
        train_log = 'Train() Summary Epoch: [{0}]\tAverage Loss {loss:.3f}\tAverage CER {cer:.3f}%\tAverage WER {wer:.3f}%'.format(
                    epoch + 1, loss=train_loss, cer=train_cer, wer=train_wer)
        #print(train_log)

        # validation
        val_loss, val_cer, val_wer = evaluate(args.train_name,model, validationLoader, criterion, device, save_output=True)
        val_log = 'Val() Summary Epoch: [{0}]\tAverage Loss {loss:.3f}\tAverage CER {cer:.3f}%\tAverage WER {wer:.3f}%'.format(
                    epoch + 1, loss=val_loss, cer=val_cer, wer=val_wer)
        #print(val_log)

        #Save current state
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_cer' : best_cer,
            'best_wer' : best_wer, 
            'Epochs'   : epoch     
        }
        torch.save(state, args.save_folder+"/"+args.train_name+'_lastEpoch.pth') 

        # in the case of the lowest WER (= the best performance), save the best model
        if best_wer > val_wer:
            val_log += "*"
            print("Found better validated model, saving to %s" % args.save_folder)
            torch.save(state, args.save_folder+"/"+args.train_name+'_Best.pth') 
            best_wer = val_wer

        print("Shuffling batches...")
        train_sampler.shuffle(epoch)

        # Learning Rate
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        Lr_inf0 = 'Learning rate (Cosine): {lr:.6f}'.format(lr=lr)
        #print(Lr_inf0)

        # Time
        Train_time_sofar = time.time() - train_start
        Taken_hour = Train_time_sofar//3600
        Taken_min = (Train_time_sofar - Taken_hour*3600)//60
        Time_info = "{}(h) {}(m) Taken".format(Taken_hour, Taken_min )
        print(Time_info)
        
        # Log Summary
        with open('./log/'+args.train_name+"_Summary.txt", "a") as ff:
            ff.write(Lr_inf0 + "\n")
            ff.write(train_log + "\n")
            ff.write(val_log + "\n")
            ff.write(Time_info + "\n\n")


if __name__ == "__main__":
    main()
