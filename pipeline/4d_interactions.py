from typing import Literal
import numpy as np
from sensitivity import VarSensitivity
from pathlib import Path
import time
import pandas as pd
import pickle
import json
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from pycirclize import Circos
from pycirclize.utils import ColorCycler
from itertools import combinations, product, islice
from scipy.interpolate import griddata
import shap
from collections import Counter
from matplotlib.patches import Patch
from pipeline.config import set_default, ggcms_only as ggcms, kg_desc

crop = 'corn'
scenario = 'rf'
target_dir = Path(r'P:\esmscratch\ANFOS_Main\ModelSensitivity\preliminary_results') / f'sensitivities_{crop}_set1' / scenario
features = set_default

corr_thresh = 0.10

feat_names_short = {
    'rsds_sum_gs': 'RAD',
    'tasmax_av_gs': 'TMX',
    'tasmin_av_gs': 'TMN',
    'pr_sum_gs': 'PR',
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
    'sand': 'SAND',
    'silt': 'SILT',
    'oc': 'OC',
    'awc': 'AWC',
}

(target_dir / 'interaction').mkdir(exist_ok=True)

with open(target_dir / 'results.p', 'rb') as f:
    results = pickle.load(f)

def shap_logodds_to_prob(shaps_log, expected_val):
    values_log = expected_val + shaps_log
    ee = np.exp(expected_val)
    expected_p = ee / (1 + ee)  # \in [0, 1]
    evals = np.exp(values_log)
    values_p = evals / (1 + evals)  # \in [0, 1]
    return values_p - expected_p

res = {}

# writer = pd.ExcelWriter(target_dir / 'interaction' / 'corrs.xlsx')

feat_short = [feat_names_short[x] for x in features]
feat_idx = [x[0] + '×' + x[1] for x in product(feat_short, feat_short)]

shaps_all = {}

for ggcm in ggcms:
    print(ggcm)
    for kg in 'ABCD':
        fig = None
        i = 0
        fig_i = 0

        shaps_p = shap_logodds_to_prob(results[ggcm][kg]['shap_ia'], results[ggcm][kg]['shap_ia_expected'])
        # shaps_all[(ggcm, kg, 'facilitating')] = np.quantile(np.maximum(np.triu(shaps_p, 1), 0), 0.75, axis=0).flatten()
        # shaps_all[(ggcm, kg, 'mitigating')] = np.quantile(np.minimum(np.triu(shaps_p, 1), 0), 0.25, axis=0).flatten()
        shaps_all[(ggcm, kg, 'facilitating')] = np.quantile(np.triu(shaps_p, 1), 0.75, axis=0).flatten()
        shaps_all[(ggcm, kg, 'mitigating')] = np.quantile(np.triu(shaps_p, 1), 0.25, axis=0).flatten()

        # Count winning IA for each sample
        sh_triu = np.triu(shaps_p, 1)
        # counts = np.zeros_like(sh_triu)
        counts_fac = np.zeros_like(sh_triu)
        counts_mit = np.zeros_like(sh_triu)
        
        for i in range(sh_triu.shape[0]):
            # counts[i, *np.unravel_index(sh_triu[i].argmax(), sh_triu.shape[1:])] += 1
            sh_fac = np.maximum(sh_triu[i], 0)
            sh_mit = np.minimum(sh_triu[i], 0)

            if np.any(sh_fac > 0):
                counts_fac[i, *np.unravel_index(sh_fac.argmax(), sh_triu.shape[1:])] += 1
            if np.any(sh_mit < 0):
                counts_mit[i, *np.unravel_index(sh_mit.argmin(), sh_triu.shape[1:])] += 1
            
        corrs = results[ggcm][kg]['corr'].values
        # score = counts.sum(axis=0).astype(int)
        # score *= (corrs.abs().values < corr_thresh)
        score_fac = counts_fac.sum(axis=0).astype(int)
        score_mit = counts_mit.sum(axis=0).astype(int)

        # res[(ggcm, kg)] = pd.Series(score.flatten(), index=[x[0] + '*' + x[1] for x in list(product(features, features))])
        res[(ggcm, kg, 'facilitating')] = pd.Series(score_fac.flatten(), index=feat_idx)
        res[(ggcm, kg, 'mitigating')] = pd.Series(score_mit.flatten(), index=feat_idx)
        res[(ggcm, kg, 'corr')] = pd.Series(corrs.flatten(), index=feat_idx)

        # results[ggcm][kg]['corr'].to_excel(writer, sheet_name=ggcm + '_' + kg)

# writer.close()

res = pd.DataFrame(res)
res['sum_facilitating'] = res.xs('facilitating', axis=1, level=2).sum(axis=1)
res['sum_mitigating'] = res.xs('mitigating', axis=1, level=2).sum(axis=1)
res['sum'] = res['sum_facilitating'] + res['sum_mitigating']
res['corr_mean'] = res.xs('corr', axis=1, level=2).mean(axis=1)
res.sort_values('sum', inplace=True, ascending=False)


# 
# d1.loc[(d1 != 0).all(axis=1)].T.to_csv(target_dir / 'interaction' / 'facilitating.csv', float_format='%.2f', encoding='utf-8-sig')

# d1 = res.xs('mitigating', axis=1, level=2)
# d1.loc[(d1 != 0).all(axis=1)].T.to_csv(target_dir / 'interaction' / 'mitigating.csv', float_format='%.2f', encoding='utf-8-sig')


# res.T.to_csv(target_dir / 'interaction' / 'scores.csv', float_format='%.2f', encoding='utf-8-sig')


shaps_all = pd.DataFrame(shaps_all, index=feat_idx)
shaps_all.xs('facilitating', axis=1, level=2).max(axis=1).sort_values(ascending=False)
shaps_all.xs('mitigating', axis=1, level=2).min(axis=1).sort_values(ascending=True)

d1 = shaps_all.xs('facilitating', axis=1, level=2)
(d1.loc[(d1 > 0.001).any(axis=1)] * 100).T.to_csv(target_dir / 'interaction' / 'facilitating_q3.csv', float_format='%.1f', encoding='utf-8-sig')

d2 = shaps_all.xs('mitigating', axis=1, level=2)
d2 = d2.loc[(d2 != 0).any(axis=1)]

(d2 * 100).T.to_csv(target_dir / 'interaction' / 'mitigating_q1.csv', float_format='%.1f', encoding='utf-8-sig')


shaps_all.to_csv(target_dir / 'interaction' / 'shap_ia_quantiles.csv', encoding='utf-8-sig')

# Plot
data_plot = res.drop('tasmax_av_gs×tasmin_av_gs', axis=0).iloc[:3]

fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
star_offset = 0.27

# Each Shapley data point can be associated with a facilitating (> 0) as well as mitigating (< 0) winner, so 
# the total for computing the percentage is 1000 * 2.
n_total = 2000

for i, kg in enumerate('ABCD'):
    row = i // 2
    col = i % 2

    d = data_plot.xs(kg, axis=1, level=1).melt(col_level=None, ignore_index=False).reset_index()
    d.columns = ['Interaction', 'GGCM', 'Type', 'Value']
    # d = d[d['Type'] != 'corr']

    d2 = d[d['Type'] == 'facilitating'].rename({'Value': 'Facilitating'}, axis=1)
    d2['Facilitating'] = d2['Facilitating'] / n_total * 100
    d2['Mitigating_real'] = d[d['Type'] == 'mitigating']['Value'].values / n_total * 100
    d2['Mitigating'] = d2['Mitigating_real'] + d2['Facilitating']
    # d2['corr'] = d[d['Type'] == 'corr']['Value'].values

    sns.barplot(data=d2, x='GGCM', y='Mitigating', hue='Interaction', alpha=0.5, legend=False, ax=axs[row, col])
    sns.barplot(data=d2, x='GGCM', y='Facilitating', hue='Interaction', legend=i == 0, ax=axs[row, col])

    star_x = np.repeat(np.arange(len(ggcms)), 3).astype(float)
    star_x[0::3] = star_x[0::3] - star_offset
    star_x[2::3] = star_x[2::3] + star_offset
    star_y = d2['Mitigating'].values + 1
    star_y[d[d['Type'] == 'corr']['Value'].abs().values > corr_thresh] = np.nan

    axs[row, col].plot(star_x, star_y, marker=(5, 2), markersize=5, linestyle='none', color='dimgrey')

    axs[row, col].set_title(f'{kg_desc[kg]} ({kg})')
    axs[row, col].set_ylabel(None)
    axs[row, col].set_xlabel(None)
    axs[row, col].tick_params(axis='x', labelrotation=90)

fig.supylabel(r'Percentage of samples [%]')

custom_patches = [
    Patch(facecolor='C0', alpha=0.5, label='Mitigating'), 
    Patch(facecolor='C0', alpha=1.0, label='Facilitating')
]

handles, labels = axs[0, 0].get_legend_handles_labels()
first_legend = axs[0, 0].legend(handles, labels, title='Interaction', loc='upper left')
axs[0, 0].add_artist(first_legend)  # this retains the first legend

# Now add the second, manual one
axs[0, 0].legend(handles=custom_patches, title='Contribution Type', loc='upper left',  bbox_to_anchor=(.35, 1))

fig.tight_layout()
fig.savefig(target_dir / 'interaction' / 'interactions.png', dpi=300)
fig.savefig(target_dir / 'interaction' / 'interactions.svg')

fig.show()



print('done')
