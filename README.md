
# MANNER (Multi-view Attention Network for Noise ERasement)

This is a Pytorch implementation of MANNER: Multi-view Attention Network for Noise ERasement. MANNER is a deep learning model for speech enhancement in time-domain. MANNER consists of encoder-decoder based on U-net. Each encoder and decoder contains Up/Down conv, Residual Conformer block, and Multi-view Attention block. MANNER representing full information of the signal efficiently addresses channel and long sequential features. Experimental results on the VoiceBank-DEMAND dataset suggest that MANNER achieves state-of-the-art performance. In addition, among time-domain models, MANNER shows the efficient results in terms of inference speed and memory usage.

The overall architecture of MANNER is as below:

<img src="https://user-images.githubusercontent.com/49035579/133537738-fb3ac1a4-aa9e-4c58-a820-8e9f613f118a.jpg" width="800" height="300">

# Installation & Enviornment

The OS, python and pytorch version needs as below:
- Windows
- Linux 
- python >= 3.7.4
- pytorch == 1.7.1
- torchaudio == 0.7.2

Notice that if you use windows, "tempo" and "speed" augmentation based on sox are not avaiable. You need to use "shift" augmentation.

You can install requirements through git and requirements.txt.
```C
git clone 
cd manner
pip install -r requirements.txt
```

# Prepare for usage

## 1. Prepare dataset

We use VoiceBank-DEMAND (Valentini) dataset consisting 28 speakers for training MANNER. 

- The dataset can be downloaded [here](https://datashare.ed.ac.uk/handle/10283/2791).

## 2. Downsample

The sample rate of the dataset is 48kHz.

For a fair comparison we downsample the audio files from 48kHz to 16kHz.

- To downsample the audio, run the following code and edit the directorys.

```
python downsampling.py
```
  
- In the downsampleing.py script, you should change the contents as follows.
  
```
downsample_rate = 16000
clean_train_path = 'The original clean trainset path'
noisy_train_path = 'The original noisy trainset path'
clean_test_path = 'The original clean testset path'
noisy_test_path = 'The original noisy testset path'
resample_path = 'Resampled path'
```
  
## 3. Make data path files

We make json file consisting of the audio path for loading data efficiently. Train (clean, noisy) and 
Test (clean, noisy): four json files need to be generated for training. 

Noitce that the data is downsampled.

- To make json file, run the following code and edit the directorys.

```
python make_datapath.py
```

- In the make_datapath.py script, you should change the contents as follows.

```
clean_train_path = 'The resampled clean trainset path'
noisy_train_path = 'The resampled noisy trainset path'
clean_test_path = 'The resampled clean testset path'
noisy_test_path = 'The resampled noisy testset path'
```

# How to use

## 1. Train

### Training with default settings

You can train MANNER with default setting by running the following code.

```
python main.py train
```

### Training with other arguments
If you want to edit model settings, you can run the following code with other arguments.

```
python main.py train --hidden 60 --depth 4 --growth 2 --kernel_size 8 --stride 4 --segment_len 64

MANNER arguments:
  --in_channels: initial in channel size
  --out_channels: initial out channel size
  --hidden: channel size to expand
  --depth: number of layers for encoder and decoder
  --kernel_size: kernel size for UP/DOWN conv
  --stride: stride for UP/DOWN conv
  --growth: channel expansion ration 
  --head: number of head for global attention
  --segment_len: chunk size for overlapped chunking
```

### Training with logging


## 2. evaluation

## 3. Pretrained weights



# Experimental Results

|Model|PESQ|STOI|CSIG|CBAK|CVOL|
|:---:|:---:|:---:|:---:|:---:|:---:|
|MANNER (small)|3.12|95|4.45|3.61|3.82|
|MANNER|3.21|95|4.53|3.65|3.91|


# Citation

# License
