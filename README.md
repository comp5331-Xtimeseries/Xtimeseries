# Xtimeseries
- [LSTNet](#lstnet)
  * [Paper](#paper)
  * [File structure](#file-structure)
  * [Usage](#usage)
  * [Environment](#environment)
- [TPA-LSTM Original](#tpa-lstm-original)
  * [Paper](#paper-1)
  * [Usage](#usage-1)
  * [File structure](#file-structure-1)
  * [Environment](#environment-1)
- [TPA-LSTM Pytorch Implementation](#tpa-lstm-pytorch-implementation)
  * [Usage](#usage-2)
  * [File structure](#file-structure-2)
  * [Environment](#environment-2)
- [mWDN](#mwdn)
  * [Paper](#paper-2)
  * [Usage](#usage-3)
  * [Environment](#environment-3)







## LSTNet

### Paper

Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks.(https://arxiv.org/abs/1703.07015)

This repo is forked from and modified on the authors' repo: https://github.com/laiguokun/LSTNet

### Usage
Check the `save` folder for the trained models and their configurations

In summary, we ran 
``` 
python main.py --gpu 0 --data exchange_rate --save exchange_rate.pt --hidCNN 50 --hidRNN 50 --L1Loss False --output_fun None
python main.py --gpu 0 --data electricity --save elec.pt --output_fun Linear --horizon 24
python main.py --gpu 0 --data solar --save solar_AL.pt --hidSkip 10 --output_fun Linear
python main.py --gpu 0 --data traffic --save traffic.pt --hidSkip 10
python main.py --gpu 0 --data sp500 --save sp500.pt --window 60 --horizon 6 --skip 20 --CNN_kernel 6 --hidSkip 1
```
Check `main.py` for the usage of the arguments.

### File structure

`Dataset.py`: API for accessing 5 datasets

`main.py`: training logics for LSTNet

`utils.py`: helper functions to convert data to batches

`models/LSTNet.py`: the architecture of LSTNet

`Optim.py`: learning rate decay logics for SGD  (unchanged from the original authors code)
`


### Environment 
Linux

Python 3.6 and PyTorch 1.6.0

## TPA-LSTM Original

### Paper

Original Implementation of [''Temporal Pattern Attention for Multivariate Time Series Forecasting''](https://arxiv.org/abs/1809.04206).


### Usage
Training
```
$ ./runTrainAll.sh
```
Testing
```
$ ./runTestAll.sh
```

### File structure
- The main.py excute all the training or testing
- The model and its supplementary was stored in lib

### Environment

* python3.6.6

You can check and install other dependencies in `requirements.txt`.
```
$ pip install -r requirements.txt
```
## TPA-LSTM Pytorch Implementation

The code was adapted from https://github.com/jingw2/demand_forecast
This implementation is different from the paper.


### Usage
Use either of the command to train, evaluate and get the result of the specific dataset
```
./run_elec.sh
./run_exchange.sh
./run_solar.sh
./run_traffic.sh
```

### File structure
- ./data store the downloaded dataset
- ./model store the trained model
- ./result store the training, evaluation and testing result
- main.py execute the training, evaluation and testing process
- tpaLSTM.py has the model structure  

### Environment
Please install [Pytorch](https://pytorch.org/) before run it, and

```python
pip install -r requirements.txt
```

## mWDN

### Paper

Wang, J., Wang, Z., Li, J., & Wu, J. (2018, July). Multilevel wavelet decomposition network for interpretable time series analysis. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2437-2446).

Source repo: https://github.com/timeseriesAI/tsai

### Usage
Training and testing on our dataset-of-interest:
sh run.sh


### Environment 
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch

