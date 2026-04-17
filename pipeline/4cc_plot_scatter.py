from typing import Literal
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit, least_squares
import itertools as it
import pickle
import matplotlib.pyplot as plt
import argparse


from sens_common import set_default, ggcms_only as ggcms, kg_desc
from sensitivity import Report

parser = argparse.ArgumentParser()
parser.add_argument('--results', default=None, required=True, type=Path, help='Path to results.p')
parser.add_argument('--out', default=None, required=True, type=Path, help='Output directory to store collection of plots (a subdirectory is created)')
parser.add_argument('--crop', default='corn', choices=['corn', 'soy'], type=str)
parser.add_argument('--irr', default='rf', choices=['rf', 'irr'], type=str)
args = parser.parse_args()

results_path = args.results
out_path = args.out
crop = args.crop
irr = args.irr

with open(results_path, 'rb') as f:
    results = pickle.load(f)



crop: Literal['corn', 'soy'] = 'corn'
irr = 'rf'
target_dir = Path(r'P:\esmscratch\ANFOS_Main\ModelSensitivity\preliminary_results') / f'sensitivities_{crop}_set1' / irr
out_dir = target_dir / 'scatter'
out_dir.mkdir(exist_ok=True)

if crop == 'soy':
    ggcms = [x for x in ggcms if x != 'DSSAT-Pythia']

features = set_default

with open(target_dir / 'results.p', 'rb') as f:
    results = pickle.load(f)

colors = ['olivedrab', 'darkred', 'darkgoldenrod', 'cornflowerblue']
cols_in_plot = 4

fbounds = {}
for model_name, model_data in results.items():
    if model_name == 'IIZUMI':
        continue
    for kg_name, kg_data in model_data.items():
        if kg_name == 'E':
            continue
        shaps = kg_data['shap']
        
        if kg_name in fbounds:
            lb, ub = fbounds[kg_name]
        else:
            lb = np.ones(len(features)) * np.inf
            ub = np.ones(len(features)) * -np.inf
        lb = np.min([lb, shaps.data.min(axis=0)], axis=0)
        ub = np.max([ub, shaps.data.max(axis=0)], axis=0)
        fbounds[kg_name] = (lb, ub)

def get_colors(labels):
    counts = labels.value_counts()
    groups = labels.sort_values().apply(lambda x: x if counts[x] > 1 else 0)
    c = it.count(1)
    cmap = groups.groupby(groups[groups > 0]).apply(lambda x: colors[next(c)])
    cmap[0] = 'dimgray'
    return groups.apply(lambda x: cmap[int(x)]).to_dict()

labels = pd.read_csv(target_dir / 'cluster' / f'labels.csv', index_col=0, header=[0, 1])

for i, feat in enumerate(['pr_sum_gs', 'rsds_sum_gs', 'tasmax_av_gs', 'tasmin_av_gs']):

    print(feat)
    for kg in 'ABCD':
        
        cmap = get_colors(labels[feat][kg])
        num_rows = int(np.ceil(len(ggcms) / cols_in_plot))
        fig_height = num_rows * 3.5

        plt.rcParams.update({'font.size': 14})
        fig, axs = plt.subplots(num_rows, cols_in_plot, figsize=(19.2 ,  fig_height), layout='compressed', squeeze=False)

        for i, ggcm in enumerate(ggcms):
            row = i // cols_in_plot
            col = i % cols_in_plot

            lb = fbounds[kg][0][features.index(feat)]
            ub = fbounds[kg][1][features.index(feat)]

            shap_values = results[ggcm][kg]['shap'][:, features.index(feat)]
            axs[row, col].scatter(shap_values.data, shap_values.values, s=1, alpha=0.5, color=cmap[ggcm], label=kg if i == 0 else None)
            axs[row, col].set_xlabel(Report.feat_names[feat])
            axs[row, col].set_ylim(lb, ub)
            axs[row, col].set_ylabel('Shapley value')
            axs[row, col].set_ylim(-0.5, 1)

            minor_ticks = np.linspace(lb, ub, 20)
            axs[row, col].set_xticks(minor_ticks, minor=True)
            axs[row, col].grid(axis='x', which='minor', alpha=0.5)

            axs[row, col].set_title(f'{ggcm}')

        for ax in axs.flatten():
            if not ax.has_data():
                ax.axis('off') 

        fig.suptitle(f'{kg.lower()}) Shapley values vs. {Report.feat_names[feat]}, {kg_desc[kg]} ({kg})')
        fig.savefig(out_dir / f'{feat}_{kg}.png', dpi=150)
        fig.savefig(out_dir / f'{feat}_{kg}.svg')
        plt.close()

print('done')
