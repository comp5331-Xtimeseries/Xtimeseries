import requests
import pandas as pd
import gzip
import os

class Datasets():
  """
  An abstract class.
  """
  def __getitem__(self, index):
    raise NotImplementedError

class DownloadableDatasets(Datasets):
  """
  All subclasses should specify `source` and `data_path`.

  """
  def __init__(self):
    if not os.path.exists(self.data_path):
      self.download()
    self.data=self.load()
  def __len__(self):
    return self.data.shape[1]
  def __getitem__(self, index):
    return self.data[:,index]
  def download(self):
    os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
    r = requests.get(self.source, allow_redirects=True)
    open(self.data_path, 'wb').write(r.content)
  def load(self):
    features_train=None
    with gzip.open(self.data_path) as f:
        features_train = pd.read_csv(f,header=None)
    return features_train.values

class Electricity(DownloadableDatasets):
  """
  This dataset is retrieved from 
  https://github.com/laiguokun/multivariate-time-series-data
  It records the hourly electricity consumptions of 321 clients from 2012 to 2014.

  Usage:
    >>> dataset = Electricity()
    >>> print(len(dataset))
    321
    >>> print(dataset[0].shape)
    (26304,)
  """
  source='https://github.com/laiguokun/multivariate-time-series-data/raw/master/electricity/electricity.txt.gz'
  data_path='./data/electricity.txt.gz'

class ExchangeRate(DownloadableDatasets):
  """
  This dataset consists of the daily exchange rates between the USA and eight 
  countries, including Aus-tralia, British, Canada, Switzerland, China, Japan, 
  New Zealand, and Singapore ranging from 1990 to 2016.The daily exchange rates 
  of each of the 8 countries is treated as a time series of 7,588 timesteps.
  
  Usage:
    >>> dataset = ExchangeRate()
    >>> print(len(dataset))
    8
    >>> print(dataset[0].shape)
    (7588,)
  """
  source='https://github.com/laiguokun/multivariate-time-series-data/raw/master/exchange_rate/exchange_rate.txt.gz'
  data_path='./data/exchange_rate.txt.gz'

class Solar(DownloadableDatasets):
  """
  This dataset consists of the solar power production records in the year of 
  2006. It is sampled every 10 minutes from 137 PV plants in Alabama State. Each 
  production record of the 137 PVplants is treated as a time series with 52,560 
  time steps.

  Usage:
    >>> dataset = Solar()
    >>> print(len(dataset))
    137
    >>> print(dataset[0].shape)
    (52560,)
  """
  source='https://github.com/laiguokun/multivariate-time-series-data/raw/master/solar-energy/solar_AL.txt.gz'
  data_path='./data/solar_AL.txt.gz'

class Traffic(DownloadableDatasets):
  """
  This dataset consists of a collection of 48 months (2015-2016) hourly data 
  from the California Departmentof Transportation. The data describes the road 
  occupancy rates (between 0 and 1) measured by differentsensors on the 
  San Francisco Bay area freeways. The reading of each of the 862 sensors is 
  treated as a time series with 17,544 time steps.

  Usage:
    >>> dataset = Traffic()
    >>> print(len(dataset))
    862
    >>> print(dataset[0].shape)
    (17544,)
  """
  source='https://github.com/laiguokun/multivariate-time-series-data/raw/master/traffic/traffic.txt.gz'
  data_path='./data/traffic.txt.gz'

class SP500(DownloadableDatasets):
  """
  This dataset consists of the daily closing stock prices of S&P500 stocks from 
  2013 to 2018. The dataset contains the closing price and the volumn of the 
  470 stocks listed in the S&P500 index within the entire timespan. The daily 
  closing stock prices (First 470 rows) and volumne (the last 470 rows) of each 
  of the 505 stocks is treated as a time series of 1259 timesteps. 
  This dataset is extracted from Kaggle [https://www.kaggle.com/camnugent/sandp500].

  Usage:
    >>> dataset = SP500()
    >>> print(len(dataset))
    940
    >>> print(dataset[0].shape)
    (1259,)
    >>> dataset = SP500(include_Volume=False)
    >>> print(len(dataset))
    470
    >>> print(dataset[0].shape)
    (1259,)
  """
  # private repo should be not inaccessible, please download the file directly
  source='https://github.com/cpwan/Xtimeseries/blob/main/data/S%26P500.csv.gz'
  data_path='./data/S&P500.csv.gz'
  def __init__(self,include_Volume=False):
    super(SP500,self).__init__()
    if not include_Volume:
      self.data=self.data[:,:470]
      
  def load(self):
    features_train=None
    with gzip.open(self.data_path) as f:
        features_train = pd.read_csv(f,header=[0,1],index_col=[0])
    return features_train.values
