import shap
import torch
import pandas as pd
from utils import *;
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='SHAP calculation and heatmap drawing.')
parser.add_argument('-m', '--model', 
                    help='LSTNet model.')

parser.add_argument('-d', '--data', 
                    help='Input dataset.')

parser.add_argument('-o', '--output', 
                    help='Output heatmap file.')

args = parser.parse_args()
model_path = str(args.model)
data_path = str(args.data)
outfile = str(args.output)

model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()
df = Data_utility(data_path, 0.6, 0.2, False, 24, 24 * 7, 2)

e = shap.DeepExplainer(model, df.valid[0][:10])
shap_values = e.shap_values(df.valid[0][:10])

ax = {}

f, (ax[0], ax[1], ax[2], ax[3], ax[4], ax[5], ax[6], ax[7]) = plt.subplots(8, 1, figsize=(15, 15))

for c in range(df.train[0][:10].shape[2]):
    shap_df = []
    for i in range(10):
        tmp_array = []
        for j in range(df.train[0][:10].shape[2]):
            val = 0
            for k in range(df.train[0][:10].shape[1]):
                val += abs(shap_values[c][i][k][j])
            tmp_array.append(val)
        shap_df.append(tmp_array)
    shap_df = pd.DataFrame(shap_df)
    sns.heatmap(shap_df, cmap='gray_r', ax=ax[c], )
    ax[c].set_ylabel('currency '+str(c))
    if c==7:
        ax[c].set_xlabel('features (currency)')
plt.savefig(outfile, bbox_inches='tight')