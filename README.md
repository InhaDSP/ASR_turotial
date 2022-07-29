# Digital Speech Processing : Automatic Speech Recoginition with LibriSpeech dataset

## How to train
    # Set the environment
    0. Virtual environment (ex: conda) is recommended 
    1. ' pip install -r requirements.txt' or install : numpy, pandas, python-Levenshtein, librosa, numba, matplotlib
    2. Install pytorch considering your CUDA version : https://pytorch.org/get-started/locally/
    3. ' unzip Libri_data.zip -d ./DSP_project/ ' # unzip dataset
    4. Set PATHs in train.py and test.py (+ set data path at ./Libri_Data/*.csv files)
    
    # train and test
    5. ' python DSP_ASR_LibriSpeech/train.py '
    6. ' python DSP_ASR_LibriSpeech/test.py '  

## Recommended Environmet
- Ubuntu 20.04.3 LTS
- CUDA Version: 11.5 
- Python 3.8

## Dataset
: Part of Libri Speech Clean data 360 (label length is under 150)
    - Download it from online

## Base Model
: Location based Attention 
    - CER 36.318%  WER 73.181% at 160 Epochs
    - CER 31.254%  WER 66.985% at 200 Epochs

## Performance metric 
: Word Error Rate = (S + I + D) / N
    - S : # of substitutions
    - I : # of deletions
    - D : # of insertions
    - N : # of words in the target label


## Reference
- specaugmentation code : https://github.com/SeanNaren/deepspeech.pytorch
- Model (CLOVA Call) : https://github.com/clovaai/ClovaCall

## For Beginners
### Keywords, good to study
- Listen, attend and spell
- SpecAugment
- Learning scheduler
- teacher forcing

### Good reference
- Paperwithcode for LibriSpeech https://paperswithcode.com/sota/speech-recognition-on-librispeech-test-clean

## For students taking DSP class
### submit as zip file, including .. 
- All code files you uesd for training
- One best trained model(.pth) and test reult 
- Report as pdf (detail will be announced)
