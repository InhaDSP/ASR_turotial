# Ditial Speech Processing : Automatic Speech Recoginition with libri Speech dataset

## How to train
    # setting
    0. virtual environment is recommended 
    1. ' pip install -r requirements.txt' or install : numpy, pandas, python-Levenshtein, librosa, numba, matplotlib
    2. intall pytorch considering your CUDA version : https://pytorch.org/get-started/locally/
    3. ' unzip Libri_data.zip -d ./DSP_project/ ' # unzip dataset
    4. setting PATHs in train.py and test.py (+ set data path at ./Libri_Data/*.csv files)
    
    # train and test
    5. ' python DSP_ASR_LibriSpeech/train.py '
    6. ' python DSP_ASR_LibriSpeech/test.py '  

## Environmet
    Ubuntu 20.04.3 LTS
    CUDA Version: 11.5 
    python 3.8

## Dataset
: part of Libri Speech Clean data 360 (label length is under 150)
    - download it from online.

## Base Model
: Location based Attention 
    - CER 36.318%  WER 73.181% at 160 Epochs
    - CER 31.254%  WER 66.985% at 200 Epochs

## performance metric 
: Word Error Rate = (S + I + D) / N
    - S : # of substitutions
    - I : # of deletions
    - D : # of insertions
    - N : # of words in the target label


## Reference
- specaugmentation code : https://github.com/SeanNaren/deepspeech.pytorch
- Model (Clova Call) : https://github.com/clovaai/ClovaCall

## For Beginners
### key words, good to study
- Listen, attend and spell
- SpecAugment
- Learning scheduler
- teacher forcing

### good reference
- Paperwithcode for LibriSpeech https://paperswithcode.com/sota/speech-recognition-on-librispeech-test-clean

## For students taking DSP class
### submit as zip file, including .. 
- all code files you uesd for training
- one best trained model(.pth) and test reult 
- Reoport as pdf (detail will be announced)
