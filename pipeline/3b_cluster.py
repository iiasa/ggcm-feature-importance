"""
Module to perform results clustering.
"""

import numpy as np
from pathlib import Path
import pandas as pd
import pickle
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import set_default, kg_desc
from plots import feat_names

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import ot


parser = argparse.ArgumentParser()
parser.add_argument('--results', default=None, required=True, type=Path, help='Path to results.p')
parser.add_argument('--out', default=None, required=True, type=Path, help='Output folder; a cluster/ subfolder will be created')
parser.add_argument('--crop', default='corn', choices=['corn', 'soy'], type=str)
parser.add_argument('--irr', default='rf', choices=['rf', 'irr'], type=str)
args = parser.parse_args()

results_path = args.results
out_path = args.out
crop = args.crop
irr = args.irr


with open(results_path, 'rb') as f:
    results = pickle.load(f)

if len(results) < 2:
    raise ValueError('Clustering can only be performed for 2+ GGCMs')

ggcms = list(results.keys())
features_all = set_default
features_clust = ['pr_sum_gs', 'rsds_sum_gs', 'tasmax_av_gs', 'tasmin_av_gs']
cluster_dir_name = 'cluster'

np.random.seed(42)

# Find feature bounds per KG region
fbounds = {}
for model_name, model_data in results.items():
    for kg_name, kg_data in model_data.items():
        if kg_name == 'E':
            continue
        shaps = kg_data['shap']
        
        if kg_name in fbounds:
            lb, ub = fbounds[kg_name]
        else:
            lb = np.ones(len(features_all)) * np.inf
            ub = np.ones(len(features_all)) * -np.inf
        lb = np.min([lb, shaps.data.min(axis=0)], axis=0)
        ub = np.max([ub, shaps.data.max(axis=0)], axis=0)
        fbounds[kg_name] = (lb, ub)

def wasserstein_2d(Xi, Xj):
    n, m = Xi.shape[0], Xj.shape[0]
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = ot.dist(Xi, Xj, metric='seuclidean')
    return ot.emd2(a, b, M)

def scale(x, lb, ub):
    return ((x - lb) * 2) / (ub - lb) - 1

plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(4, 4, figsize=(19.2, 20), sharex=False, sharey=True)

panel_labels = 'abcdefghijklmnop'
groups = {}
maxy = 0
for row, feature in enumerate(features_clust):
    
    print(f'Clustering "{feature}"')
    feature_idx = features_all.index(feature)
    axs[row, 0].set_ylabel(feat_names[feature])
    
    for col, kg_name in tqdm(enumerate('ABCD'), 'Iterating regions', total=4):

        dists = []
        for model_name, model_data in results.items():
            fshaps = model_data[kg_name]['shap'][:, feature_idx]
            dists.append(fshaps)

        n = len(dists)
        dist_matrix = np.zeros((n, n))

        lb = fbounds[kg_name][0][feature_idx]
        ub = fbounds[kg_name][1][feature_idx]

        for i in range(n):
            for j in range(i + 1, n):
                a = np.vstack([dists[i].values, dists[i].data]).T
                b = np.vstack([dists[j].values, dists[j].data]).T
                d = wasserstein_2d(a, b)
                dist_matrix[i, j] = dist_matrix[j, i] = d

        condensed_dist = squareform(dist_matrix)
        linkage_matrix = linkage(condensed_dist, method='average', optimal_ordering=True)

        color_thresh = 0.5
        labels = fcluster(linkage_matrix, t=color_thresh, criterion='distance')

        with plt.rc_context({'lines.linewidth': 1.5}):
            dendrogram(linkage_matrix, labels=ggcms, leaf_rotation=90, color_threshold=color_thresh, above_threshold_color='gray', ax=axs[row, col])

        if row == 0:
            axs[row, col].set_title(f'{kg_desc[kg_name]} ({kg_name})')

        axs[row, col].text(0.98, 0.97, panel_labels[row * 4 + col] + ')',
            transform=axs[row, col].transAxes,
            fontsize=16,
            verticalalignment='top',
            horizontalalignment='right')

        maxy = max(maxy, linkage_matrix[:, 2].max())

        groups[(feature, kg_name)] = dict(zip(ggcms, labels))

for ax in axs.flatten():
    ax.set_ylim(0, maxy + int(maxy * 0.01))

# Create out dir
(out_path / cluster_dir_name).mkdir(exist_ok=True)

# Store cluster labels
pd.DataFrame(groups).to_csv(out_path / cluster_dir_name / f'labels.csv')

# Store figure
fig.supylabel('Wasserstein Distance', x=0.01)
fig.tight_layout()
fig.savefig(out_path / cluster_dir_name / f'shap_clusters.svg')

print('done')
