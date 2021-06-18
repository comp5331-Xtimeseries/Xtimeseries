# SHAP for Visual Explanations of Time Series Models
## Report
[Final report](https://github.com/comp5331-Xtimeseries/Xtimeseries/blob/main/COMP_5331_Final_Report.pdf)
[Slides] (https://github.com/comp5331-Xtimeseries/Xtimeseries/blob/main/slides.pdf)
## Code structure
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
    + [File structure](#file-structure-3)
    + [Environment](#environment-3)
- [SHAP explanation](#shap-explanation)
  * [LSTNet](#lstnet-1)
    + [Usage](#usage-4)
    + [File structure](#file-structure-4)
    + [Environment](#environment-4)
  * [mWDN](#mwdn-1)
    + [Usage](#usage-5)
    + [File structure](#file-structure-5)
    + [Environment](#environment-5)








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
Training and testing on our dataset-of-interest:
```
sh run.sh
```
### File structure
`data/`: It contains all dataset-of-interest.\
`log.*.txt`: They are log files that contain all output of training.\
`main.py`: It is the main function for training.\
`model/`: It contains all trained models.\
`models/`: It contains all files to run wMDN models correctly.\
`models/InceptionTime.py`: Inception is the classifier fed with decompositions collected from mWDN framewrok.\
`models/layers.py`: It contains the implementation of building blocks of Inception.\
`models/mWDN.py`: It is the implementation of mWDN.\
`models/utils.py`: It contains the implementation of "feeding mWDN decomposed sub-series to a classifier".\
`Optim.py`: It is a wrapper of all supported optimizer.\
`run.sh`: The script to train the model and get the results we reported.\
`utils.py`: It is how we organize the dataset.

### Environment 
Operating system\
DISTRIB_ID=Ubuntu\
DISTRIB_RELEASE=18.10\
DISTRIB_CODENAME=cosmic\
DISTRIB_DESCRIPTION="Ubuntu 18.10"

Running environment\
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

`shap_heatmap.py` The script calculates and generates the heatmaps of a model and the corresponding dataset.

### Environment 

macOS


## mWDN

### Usage

First, move to the `mWDN` subfolder.

Open `mWDN_SHAP.ipynb` with jupyter notebook.

### File structure

`mWDN_SHAP.ipynb` does the following:
1. Make a /stats folder
2. Run the group of cells under "Libraries"
3. Run one group of cells under "mWDN Exchange Rate", "mWDN Solar", "mWDN Electricity", "mWDN Traffic", depending on the desired model to visualize
4. Run the group of cells under "Visualization"
5. Results are saved in the /stats folder
### Environment 

Linux
