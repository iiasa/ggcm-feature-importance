import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pipeline.config import ggcms, format_model_name, kg_desc, scen_desc
from report import Report

crop = 'corn'
irr = 'rf'
data_dir = Path(r'P:\esmscratch\ANFOS_Main\ModelSensitivity\preliminary_results') / f'sensitivities_{crop}_set1' / irr

if crop == 'soy':
    ggcms = [x for x in ggcms if x != 'DSSAT-Pythia']

data_stats = pd.read_csv(data_dir / 'stats.csv', header=[0, 1], index_col=0)
data = data_stats.loc[['anomaly_frac']].melt().rename(columns={'variable_0': 'model', 'variable_1': 'kg'}).pivot(index='model', columns='kg', values='value')
data_areas = data_stats.loc[['area_frac']].melt().rename(columns={'variable_0': 'model', 'variable_1': 'kg'}).pivot(index='model', columns='kg', values='value')
data_areas = data_areas.mean(axis=0)

data = data * 100
# data.drop('E', axis=1, inplace=True)
data.fillna(0, inplace=True)


np.random.seed(1)
fig, ax = plt.subplots(figsize=(10, 7), layout='compressed', dpi=150)

ax.boxplot(
    data, 
    labels=[f'{x} ({kg_desc[x]}, {data_areas[x]*100:.2f}%)' for x in data.columns], 
    boxprops={'color': 'grey', 'alpha': 0.4}, 
    whiskerprops={'color': 'grey'},
    capprops={'color': 'grey'},
    showfliers=False
)

i = 0
legend = []
for model, marker, color in [(g, Report.ggcm_markers[g], Report.ggcm_colors[g]) for g in ggcms]:
    xpos = np.arange(1, data.shape[1] + 1) + np.random.normal(0, 0.03)
    ax.scatter(xpos, data.loc[model], color=color, marker=marker, zorder=99)
    if model == 'IIZUMI':
        ax.scatter(xpos, data.loc[model], s=130, color=color, facecolor='none', zorder=99)

    legend.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, markersize=10, label=format_model_name(model)))
    i += 1

ax.set_ylabel('Proportion of anomalies [%]')
ax.legend(handles=legend, loc='upper left')
ax.set_xlabel('Köppen-Geiger region')
ax.set_title(f'Proportion of yield anomalies per Köppen-Geiger region ({scen_desc[(crop, irr)]})')
ax.set_ylim(0, 55)

fig.savefig(data_dir / 'proportion_anomalies.svg')
fig.savefig(data_dir / 'proportion_anomalies.png')
