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


def compare_experiments(file1, file2, file3=None, title=None):
    out_dir = 'plots/combined/'

    #shutil.rmtree(out_dir, ignore_errors=True)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if file3 is not None:
        savepath = f'{out_dir}{file1.split("/")[-1]}_&_{file2.split("/")[-1]}_&_{file3.split("/")[-1]}'
        files = [file1, file2, file3]
        nc = 3
    else:
        savepath = f'{out_dir}{file1.split("/")[-1]}_&_{file2.split("/")[-1]}'
        files = [file1, file2]
        nc = 2

    dfs_ic = []
    dfs_fid = []
    ctr = 0
    for file in files:
        ctr += 1
        json_files = []
        for x in os.walk(file):
            for y in glob.glob(os.path.join(x[0], 'curves.json')):
                json_files.append(y)

        inception_means = []
        fids = []

        for datafile in json_files:
            data = pd.read_json(datafile, lines=True)
            x = data['inception_means'].to_numpy()[0]
            inception_means.append(x)
            fids.append(data['fids'].to_numpy()[0])

        inception_means = np.array(inception_means)
        df_ic = pd.DataFrame(inception_means).transpose().stack().reset_index()
        del df_ic['level_1']
        cat_var = []
        for i in range(len(df_ic.axes[0])):
            cat_var.append(ctr)
        df_ic['Exp'] = cat_var
        dfs_ic.append(df_ic)

        fids = np.array(fids)
        df_fid = pd.DataFrame(fids).transpose().stack().reset_index()
        del df_fid['level_1']
        df_fid['Exp'] = cat_var
        dfs_fid.append(df_fid)

    comb_ic = pd.concat(dfs_ic)
    ax1 = sns.lineplot(data=comb_ic, x='level_0', y=0, ci=90, hue='Exp', palette=sns.color_palette("tab10", n_colors=nc))
    if title is not None:
        ax1.set(title=f'IS with %90 CIs: {title}')
    ax1.set_xlabel('Iterations (x5K)')
    ax1.set_ylabel('Inception score')
    ax1.set_xlim(left=0, right=30)
    ax1.set_ylim(bottom=0, top=8)
    plt.savefig(f'{savepath}_IS.png')
    plt.clf()

    comb_fid = pd.concat(dfs_fid)
    ax2 = sns.lineplot(data=comb_fid, x='level_0', y=0, ci=90, hue='Exp', palette=sns.color_palette("tab10", n_colors=nc))
    if title is not None:
        ax2.set(title=f'FID with %90 CIs: {title}')
    ax2.set_xlabel('Iterations (x5K)')
    ax2.set_ylabel('Inception score')
    ax2.set_xlim(left=0, right=30)
    ax2.set_ylim(bottom=0, top=80)
    plt.savefig(f'{savepath}_FID.png')
    plt.clf()


def plot_experiment(filepath):
    json_files = []
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

    for file in json_files:
        data = pd.read_json(file, lines=True)
        inception_means.append(data['inception_means'].to_numpy()[0])
        inception_stds.append(data['inception_stds'].to_numpy()[0])
        fids.append(data['fids'].to_numpy()[0])

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
    ax1.set_xlim(left=0, right=30)
    ax1.set_ylim(bottom=0, top=8)
    plt.savefig(f'{savepath}_ci_IS.jpg')
    plt.clf()

    # plot FID with CIs
    df_fid = pd.DataFrame(fids).transpose().stack().reset_index()
    del df_fid['level_1']
    ax2 = sns.lineplot(data=df_fid, x='level_0', y=0, ci=90)
    ax2.set(title=f'FID with %90 CIs: {title}')
    ax2.set_xlabel('Iterations (x5K)')
    ax2.set_ylabel('FID')
    ax2.set_xlim(left=0, right=30)
    ax2.set_ylim(bottom=0, top=80)
    plt.savefig(f'{savepath}_ci_FID.jpg')
    plt.clf()

    # plot individual IS experiments
    df = pd.DataFrame(inception_means).transpose()
    ax3 = sns.lineplot(data=df)
    ax3.set(title=f'Individual IS trials: {title}')
    ax3.set_xlabel('Iterations (x5K)')
    ax3.set_ylabel('Inception score')
    ax3.set_xlim(left=0, right=30)
    ax3.set_ylim(bottom=0, top=8)
    plt.savefig(f'{savepath}_ind_IS.jpg')
    plt.clf()

    # plot individual FID experiments
    df = pd.DataFrame(fids).transpose()
    ax4 = sns.lineplot(data=df)
    ax4.set(title=f'Individual FID trials: {title}')
    ax4.set_xlabel('Iterations (x5K)')
    ax4.set_ylabel('FID')
    ax4.set_xlim(left=0, right=30)
    ax4.set_ylim(bottom=0, top=80)
    plt.savefig(f'{savepath}_ind_FID.jpg')
    plt.clf()


filepath = 'results/8-4/type_lola_iter150000_bs128_lrD0.001_lrG0.001_ee5000_zerosum__top_1.0_0.2_10_8-4'
plot_experiment(filepath)

compare_experiments('results/8-4/type_sgd_etaNA_iter300000_bs128_lrD0.001_lrG0.001_ee5000',
                    'results/8-4/type_lola_iter300000_bs128_lrD0.001_lrG0.001_ee5000_top_0.1_0.2_10',
                    title="SGD vs LOLA, small net")

