#%%
import numpy as np
import pandas as pd
import json
import glob
import os
import scipy.stats as st
from matplotlib import pyplot as plt
import seaborn as sns
import shutil

json_files = []
filepath = 'results/8-4/type_lola_iter300000_bs128_lrD0.001_lrG0.001_ee5000_top_0.1_0.2_10'
out_dir = f'plots/{filepath.split("/")[1]}/{filepath.split("/")[-1]}/'

shutil.rmtree(out_dir, ignore_errors=True)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
savepath = f'{out_dir}{filepath.split("/")[-1]}'


model = filepath.split('/')[1]
if model == '16-8':
    model_size = 'large model'
elif model == '8-4':
    model_size = 'small model'
type = filepath.split('_')[1]
method = filepath.split('_')[-4]
alpha = filepath.split('_')[-3]
if method == 'top':
    top_perc = filepath.split('_')[-2]
    title = f'{model_size}, {type}, method: top ({top_perc}), α: {alpha}'
else:
    top_perc = filepath.split('_')[-2]
    title = f'{model_size}, {type}, method: {method}, α: {alpha}'

for x in os.walk(filepath):
    for y in glob.glob(os.path.join(x[0], 'curves.json')):
        json_files.append(y)

inception_means = []
inception_stds = []
fids = []

dfs = []
for file in json_files:
    data = pd.read_json(file, lines=True)
    inception_means.append(data['inception_means'].to_numpy()[0])
    inception_stds.append(data['inception_stds'].to_numpy()[0])
    fids.append(data['fids'].to_numpy()[0])
    dfs.append(data)

inception_means = np.array(inception_means)
inception_stds = np.array(inception_stds)
fids = np.array(fids)

# plot IS with CIs
df_ic = pd.DataFrame(inception_means).transpose().stack().reset_index()
del df_ic['level_1']
ax1 = sns.lineplot(data=df_ic, x='level_0', y=0, ci=90)
ax1.set(title=f'IS with %90 CIs: {title}')
ax1.set_xlabel('Iterations (x5K)')
ax1.set_ylabel('Inception score')
plt.show()
plt.savefig(f'{savepath}_ci_IS.jpg')
plt.clf()

# plot FID with CIs
df_fid = pd.DataFrame(fids).transpose().stack().reset_index()
del df_fid['level_1']
ax2 = sns.lineplot(data=df_fid, x='level_0', y=0, ci=90)
ax2.set(title=f'FID with %90 CIs: {title}')
ax2.set_xlabel('Iterations (x5K)')
ax2.set_ylabel('FID')
plt.show()
plt.savefig(f'{savepath}_ci_FID.jpg')
plt.clf()

# plot individual IS experiments
df = pd.DataFrame(inception_means).transpose()
ax3 = sns.lineplot(data=df)
ax3.set(title=f'Individual IS trials: {title}')
ax3.set_xlabel('Iterations (x5K)')
ax3.set_ylabel('Inception score')
plt.show()
plt.savefig(f'{savepath}_ind_IS.jpg')
plt.clf()

# plot individual FID experiments
df = pd.DataFrame(fids).transpose()
ax4 = sns.lineplot(data=df)
ax4.set(title=f'Individual FID trials: {title}')
ax4.set_xlabel('Iterations (x5K)')
ax4.set_ylabel('FID')
plt.show()
plt.savefig(f'{savepath}_ind_FID.jpg')
plt.clf()
