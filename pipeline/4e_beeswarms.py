

from pathlib import Path
import argparse


import os
print('*********')
print(os.environ['PYTHONPATH'])
print('*********')
exit()


ncols = 4
ggcms_per_fig = 5
fig_height_per_ggcm = 5
fig_i = 1
row = 0

fig, axs = plt.subplots(ggcms_per_fig, ncols, figsize=(19.2 , fig_height_per_ggcm * ggcms_per_fig), layout='compressed')

(self.target_dir / 'beeswarm').mkdir(exist_ok=True)

for model_name, model_data in data.items():

    for col, (kg_name, kg_data) in enumerate(model_data.items()):
        
        if kg_name == 'E':
            continue

        stats = kg_data['stats']
        h_harv = stats['area'] / 1e6
        p_harv = stats['area_frac'] * 100
        p_ano = stats['anomaly_frac'] * 100
        p_misc = stats['misc_rate'] * 100
        p_recall = stats['recall'] * 100
        p_precision = stats['precision'] * 100
        p_auroc = stats['auroc'] * 100
        
        if len(kg_data['shap'].shape) == 3 and kg_data['shap'].shape[2] == 2:
            kg_data['shap'] = kg_data['shap'][:, :, 0]
            kg_data['shap'].output_dims = ()
            kg_data['shap'].output_names = None
            kg_data['shap'].output_indices = None

        kg_data['shap'].feature_names = [self.feat_names[x.lstrip('*')] if x.lstrip('*') in self.feat_names else x for x in kg_data['shap'].feature_names]

        shap.plots.beeswarm(
            kg_data['shap'], 
            max_display=len(self.features), 
            order=kg_data['shap'].abs.mean(0), 
            show=False,
            color_bar=False, 
            ax=axs[row, col],
            plot_size=None
        )
        
        if row == 0:
            axs[row, col].set_title(f'{kg_desc[kg_name]} ({kg_name})')

        axs[row, col].set_xlabel(None)
        axs[row, col].set_xlim(-1, 1)
        axs[row, col].annotate(
            f'{h_harv:.2f} Mha ({p_harv:.2f}%), {p_ano:.1f}% ano., {p_auroc:.1f}% AUROC',
            xy=(0, 0), xytext=(100, -25), xycoords='axes fraction', textcoords='offset points', va='top', ha='center', size=7)         

        if col == 0:
            axs[row, col].set_ylabel(model_name)

    if row > 0 and (row + 1) % ggcms_per_fig == 0:
        fig.savefig(self.target_dir / 'beeswarm' / f'shaps_{fig_i}.png')
        fig.savefig(self.target_dir / 'beeswarm' / f'shaps_{fig_i}.svg')
        nrows = min(len(data) - fig_i * ggcms_per_fig, ggcms_per_fig)
        fig_i += 1
        row = 0
        fig, axs = plt.subplots(nrows, ncols, figsize=(19.2 , fig_height_per_ggcm * nrows), layout='compressed')

    else:
        row += 1

if row % ggcms_per_fig != 0:
    fig.savefig(self.target_dir / 'beeswarm' / f'shaps_{fig_i}.png')
    fig.savefig(self.target_dir / 'beeswarm' / f'shaps_{fig_i}.svg')
    