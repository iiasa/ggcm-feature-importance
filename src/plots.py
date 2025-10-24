import os
from typing import List, Literal, Optional, Union
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import math
import seaborn as sns
import shap
from scipy.stats import spearmanr

from pathlib import Path
from itertools import product
import matplotlib.gridspec as gridspec

feat_names = {
    'rsds_sum_gs': 'solar radiation',
    'tasmax_av_gs': 'maximum temp.',
    'tasmin_av_gs': 'minimum temp.',
    'pr_sum_gs': 'precipitation',
    'wet_sum_gs': 'wet days',
    'dry_sum_gs': 'dry days',
    'hdd_sum_gs': 'heating degree days',
    'kdd_sum_gs': 'killing degree days',
    'frt_sum_gs': 'frost days',
    'ice_sum_gs': 'ice days',
    'r10_sum_gs': 'heavy precipitation',
    'r20_sum_gs': 'extreme precipitation',
    'cwd_sum_gs': 'consecutive wet days',
    'cdd_sum_gs': 'consecutive dry days',
    'sand': 'sand',
    'silt': 'silt',
    'oc': 'OC',
    'awc': 'AWC',
}

ggcm_markers = {
    'ACEA': 'o', 
    'CROVER': 'v', 
    'CYGMA1p74': '^', 
    'DSSAT-Pythia': '<', 
    'EPIC-IIASA': '>', 
    'ISAM': '8', 
    'LDNDC': 's', 
    'LPJ-GUESS': 'p', 
    'LPJmL': 'h', 
    'pDSSAT': 'H', 
    'PEPIC': 'D', 
    'PROMET': 'd', 
    'SIMPLACE-LINTUL5': 'P',
    'IIZUMI': 'X'
}

def plot_importance(
        data, features: List[str], top_n: Optional[int] = None, top_percentile: float = 5, regions: str = None,  
        example_model: str = None, example_region: str = None, example_feature: str = None, 
        scen_desc = None, type: Literal['normalized', 'raw_prob'] = 'normalized'):

    if top_n is None:
        top_n = len(features)

    example_feature_idx = features.index(example_feature) if example_feature is not None else 0
    res = {}
    max_shap = 0

    for model_name, model_data in data.items():
        for kg_name in regions:
            if kg_name not in model_data:
                continue

            kg_data = model_data[kg_name]
            shap_values = kg_data['shap']

            if type != 'raw_prob':
                shap_star = shap_values.values / shap_values.abs.values.sum(axis=1).repeat(shap_values.shape[1]).reshape(shap_values.shape)
            else:
                shap_star = shap_values.values

            td_len = int(shap_values.shape[0] * top_percentile * 0.01)
            td_indices = np.argpartition(shap_values.values, -td_len, axis=0)[-td_len:]  # use actual Shaps to find quantile
            td_values = shap_star[td_indices, np.arange(shap_values.shape[1])]  # use Shap* for position on x
            td_data = shap_values.data[td_indices, np.arange(shap_values.shape[1])]

            mean_values = np.mean(td_values, axis=0)  # shap values
            mean_data = np.mean(td_data, axis=0)  # feature values
            data_min = shap_values.data.min(0)
            data_max = shap_values.data.max(0)

            # Use this for normalization to conform with SHAP implementation
            data_min, data_max = _get_minmax_shap(shap_values.data)

            mean_data_norm = (mean_data - data_min) * 2 / (data_max - data_min) - 1  # normalize into [-1, 1]
            mean_data_norm[mean_data_norm > 1] = 1
            mean_data_norm[mean_data_norm < -1] = -1

            res[(kg_name, model_name, 'shapley')] = mean_values
            res[(kg_name, model_name, 'feature')] = mean_data_norm
            res[(kg_name, model_name, 'spearman')] = _spearman(shap_values.values, shap_values.data)
            res[(kg_name, model_name, 'auroc')] =  kg_data['stats']['auroc']

            max_shap = max(max_shap, mean_values.max())

            if model_name == example_model and kg_name == example_region:
                example_rect_x = td_values[:, example_feature_idx].min()
                example_rect_width = td_values[:, example_feature_idx].max() - example_rect_x

    res_all = pd.DataFrame(res, index=features).T
    res_all.index.names = ['region', 'ggcm', 'value_type']

    max_shap = 1
    ncols = 2
    nrows = 3 if len(regions) > 4 or example_feature is not None else 2

    box_font_size = 8 if top_n is None else 10
    if top_n < 10:
        fig_height = 12 if nrows <= 2 else 15
    else:
        fig_height = 15 if nrows <= 2 else 22
    
    plt.rcParams.update({'font.size': 14})

    fig = plt.figure(figsize=(19.2, fig_height), layout='compressed')

    gs0 = gridspec.GridSpec(1, 2, width_ratios=[0.95, 0.05], figure=fig)
    gs1 = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs0[0])
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[1])
    
    cmap = shap.plots.colors.red_blue

    n = 256
    n_third = n // 3
    lower = cmap(np.linspace(0, 0.33, n_third))  # Lower third: colormap 0–0.33
    middle = np.tile(np.array([[0.5, 0.5, 0.5, 1.0]]), (n_third, 1))  # Middle third: grey
    upper = cmap(np.linspace(0.66, 1.0, n - 2 * n_third))  # Upper third: colormap 0.66–1.0
    colors = np.vstack((lower, middle, upper))
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    cpick = ScalarMappable(norm=Normalize(vmin=-1,vmax=1), cmap=custom_cmap)
    cpick.set_array([])

    ggcm_ids = {v: k for k, v in enumerate(data.keys(), 1)}

    for kg_name, (row, col) in zip(regions, product(range(nrows), range(ncols))):
        ax = fig.add_subplot(gs1[row, col])
        res_kg = res_all.loc[kg_name]
        mean_shaps = res_kg.loc[(slice(None), 'shapley'), :].mean(0).sort_values(ascending=True)

        features_sorted = list(mean_shaps.index)
        if top_n is not None:
            features_sorted = features_sorted[-top_n:]
            mean_shaps = mean_shaps[-top_n:]
        res_kg = res_kg[features_sorted]
        shaps = res_kg.loc[(slice(None), 'shapley'), :].values
        feature_vals = res_kg.loc[(slice(None), 'feature'), :].values
        corrs = res_kg.loc[(slice(None), 'spearman'), :].values
        aurocs = [x[kg_name]['stats']['auroc'] for x in data.values() if kg_name in x]

        ax.set_title(kg_name)
        
        if type == 'raw_prob':
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"+{(x * 100):.0f}%"))
        
        ggcm_available = res_kg.loc[(slice(None), 'shapley'), :].index.get_level_values(0)
        ggcm_available = [k for k, v in ggcm_ids.items() if k in ggcm_available]
        _draw_axis(
            ax, features_sorted, mean_shaps, shaps, corrs, feature_vals, cpick, box_font_size, ggcm_available, aurocs, show_y2_label=col==1)
        ax.set_xlim((0, max_shap))

        if kg_name == example_region:
            rect_width = 0.02
            rect_height = 0.7
            feat_idx = features_sorted.index(example_feature)
            corr_y = corrs[list(data.keys()).index(example_model), feat_idx]
            rect_y = feat_idx + corr_y * 0.46 - 0.22
            rect_x = shaps[list(data.keys()).index(example_model), feat_idx]
            p = plt.Rectangle((rect_x - rect_width / 2, rect_y ), rect_width, rect_height, fill=False, ls=':', zorder=10)
            ax.add_patch(p)

    ax_cb = fig.add_subplot(gs2[0])
    ax_cb.set_axis_off()
    cb = fig.colorbar(cpick, ax=ax_cb, ticks=[-1, 1], aspect=200)  # ax=ax, 
    cb.set_ticklabels(['Low', 'High'])
    cb.set_label('Feature value', size=12, labelpad=0)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)

    example_ax = fig.add_subplot(gs1[nrows - 1, ncols - 1])

    # Beeswarm example
    if example_model:
        example_shap = data[example_model][example_region]['shap']
        example_shap_order = example_shap.abs.mean(0)
        example_ranks = example_shap_order.values.argsort()[::-1].argsort()
        example_shap.feature_names = [feat_names[x.lstrip('*')] if x.lstrip('*') in feat_names else x for x in example_shap.feature_names]

        # Grouping of cut off features disabled directly in shap
        shap.plots.beeswarm(
            example_shap, 
            max_display=top_n, 
            order=example_shap_order, 
            show=False, 
            color_bar=False, 
            ax=example_ax,
            plot_size=None
        )
        example_ax.set_title(f'Example: Shapley values for {example_model}, KG class {example_region}')
        example_ax.set_xlabel(None)
        example_rect_y = top_n - example_ranks[features.index(example_feature)] - 1.5

        p = plt.Rectangle((example_rect_x, example_rect_y), example_rect_width, 1, fill=False, ls=':', zorder=10)
        example_ax.add_patch(p)
        example_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{'+' if x >= 0 else '-'}{x * 100:.0f}%"))

    else:
        example_ax.set_axis_off()

    legend = []
    for model, marker in ggcm_markers.items():
        if model not in data:
            continue
        legend.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor='slategrey', markersize=10, label=model))

    example_ax.legend(handles=legend, ncol=2, fontsize='medium', loc='lower right')

    if scen_desc != None:
        fig.suptitle(f'Feature importance scores per KG region ({scen_desc})')

    if type != 'raw_prob':
        fig.supxlabel(f'Importance score')
    else:
        fig.supxlabel(f'Contribution to anomaly probability [%]')
    return fig

def _get_minmax_shap(data):
    min_ = np.zeros(data.shape[1])
    max_ = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        vmin = np.nanpercentile(data[:, i], 5, axis=0)
        vmax = np.nanpercentile(data[:, i], 95, axis=0)
        if vmin == vmax:
            vmin = np.nanpercentile(data[:, i], 1)
            vmax = np.nanpercentile(data[:, i], 99)
            if vmin == vmax:
                vmin = np.min(data[:, i])
                vmax = np.max(data[:, i])
        min_[i] = vmin
        max_[i] = vmax
    return min_, max_

def _draw_axis(
        ax, features_sorted, mean_abs_shaps, shaps, corrs, feature_vals, cpick, box_font_size, 
        ggcm_available:list, aurocs: list, show_y2_label=True):
    feat_labels = list(map(lambda x: feat_names[x.lstrip('*')] if x.lstrip('*') in feat_names else x, features_sorted))
    ax.axvline(x=0, color="#999999", zorder=-1)
    ax.set_yticks(range(len(features_sorted)), feat_labels, fontsize=13)
    marker_size = 100

    lim = (-1, len(features_sorted))
    ax.set_ylim(lim)
    ax2 = ax.twinx()
    ax2.set_ylim(lim)
    ax2.set_yticks(range(len(mean_abs_shaps)), [f'{x:.2f}' for x in mean_abs_shaps.values], fontsize=13)
    if show_y2_label:
        ax2.set_ylabel('Mean abs. SHAP value')

    for i, feat in enumerate(features_sorted):
        y = np.ones(shaps.shape[0]) * i + corrs[:, i] * 0.3
        
        ax.axhline(y=i, color="#cccccc", lw=2, dashes=(1, 5), zorder=-1)
        ax.axhline(y=i + 0.5, color="#cccccc", lw=1, zorder=-1)
        mx, my = _decollide_markers(shaps[:, i], y, r=0.01, y_scale=len(features_sorted), max_iters=100)

        for j in range(shaps.shape[0]):
            box_color = cpick.to_rgba(feature_vals[j, i], alpha=0.8)
            box_lw = 1 if ggcm_available[j] != 'IIZUMI' else 2
            marker = ggcm_markers[ggcm_available[j]]
            ax.scatter(mx[j], my[j], s=marker_size, color=box_color, marker=marker, edgecolors='white', zorder=99)

    ax.axhline(y=-0.5, color="#cccccc", lw=1, zorder=-1)

def _decollide_markers(x, y, r, x_scale=1.0, y_scale=1.0, max_iters=100, tolerance=1e-4):
    x = x.copy()
    y = y.copy()
    n = len(x)
    
    for _ in range(max_iters):
        # Normalized coordinate differences
        dx = (x[:, None] - x[None, :]) / x_scale
        dy = (y[:, None] - y[None, :]) / y_scale
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        np.fill_diagonal(dist, np.inf)

        overlap = 2 * r - dist
        mask = overlap > 0

        if not np.any(mask):
            break

        # Normalized directions
        dx_norm = dx / dist
        dy_norm = dy / dist

        # Displacement in normalized space
        disp_x_norm = np.zeros_like(dx)
        disp_y_norm = np.zeros_like(dy)
        disp_x_norm[mask] = dx_norm[mask] * (overlap[mask] / 2)
        disp_y_norm[mask] = dy_norm[mask] * (overlap[mask] / 2)

        # Convert displacements back to original units
        disp_x = disp_x_norm * x_scale
        disp_y = disp_y_norm * y_scale

        # Net movement for each point
        total_disp_x = np.sum(disp_x - disp_x.T, axis=1)
        total_disp_y = np.sum(disp_y - disp_y.T, axis=1)

        x += total_disp_x
        y += total_disp_y

        if (np.max(np.abs(total_disp_x)) < tolerance and np.max(np.abs(total_disp_y)) < tolerance):
            break

    return x, y

def _spearman(x, y):
    if x.shape != y.shape:
        raise ValueError("X and Y must have the same shape")
    corrs = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        if x[:, i].var() == 0 or y[:, i].var() == 0:
            corrs[i] = 0
        else:
            rho, _ = spearmanr(x[:, i], y[:, i])
            corrs[i] = rho
    return corrs
