- [Time series models](#time-series-models)
  * [LSTNet](#lstnet)
    + [Usage](#usage)
    + [File structure](#file-structure)
    + [Environment](#environment)
  * [TPA-LSTM Original](#tpa-lstm-original)
    + [Usage](#usage-1)
    + [File structure](#file-structure-1)
    + [Environment](#environment-1)
  * [TPA-LSTM Pytorch Implementation](#tpa-lstm-pytorch-implementation)
    + [Usage](#usage-2)
    + [File structure](#file-structure-2)
    + [Environment](#environment-2)
  * [mWDN](#mwdn)
    + [Usage](#usage-3)
    + [Environment](#environment-3)
- [SHAP explanation](#shap-explanation)
  * [LSTNet](#lstnet-1)
    + [Usage](#usage-4)
    + [File structure](#file-structure-3)
    + [Environment](#environment-4)






# Time series models

## LSTNet

### Usage

First, move to the `LSTNet` subfolder.

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

### Usage

First, move to the `TPALSTM-Origional` subfolder.


Training
```
$ ./runTrainAll.sh
```
Testing
```
$ ./runTestAll.sh
```

### File structure
- The `main.py` excute all the training or testing
- The model and its supplementary was stored in `lib`

### Environment

* python3.6.6

You can check and install other dependencies in `requirements.txt`.
```
pip install -r requirements.txt
```
## TPA-LSTM Pytorch Implementation

### Usage

First, move to the `TPA-LSTM-NotOrigional` subfolder.


Use either of the command to train, evaluate and get the result of the specific dataset
```
./run_elec.sh
./run_exchange.sh
./run_solar.sh
./run_traffic.sh
```

### File structure
- `./data` store the downloaded dataset
- `./model` store the trained model
- `./result` store the training, evaluation and testing result
- `main.py` execute the training, evaluation and testing process
- `tpaLSTM.py` has the model structure  

### Environment
Please install [Pytorch](https://pytorch.org/) before run it, and

```python
pip install -r requirements.txt
```

## mWDN

### Usage

First, move to the `mWDN` subfolder.


Training and testing on our dataset-of-interest:
```
sh run.sh
```

### Environment 
```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

# SHAP explanation

## LSTNet

### Usage

First, move to the `LSTNet SHAP` subfolder.

an example to show how to run the program
```
python3 shap_heatmap.py -m exchange_rate.pt -d exchange_rate.txt -o heatmap.pdf
```

### File structure

The script calculates and generates the heatmaps of a model and the corresponding dataset.

### Environment 

macOS
