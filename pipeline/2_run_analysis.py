"""
Main analysis module for assembling metamodel training data and deriving SHAP values, correlations, etc.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import pickle
import argparse

from config import set_default, set_extreme
from attribution import Analysis


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default=None, required=True, type=Path, help='Directory containing processed climate and yield data')
parser.add_argument('--inputdir', default=None, required=True, type=Path, help='Directory containing auxilary data')
parser.add_argument('--out', default=None, required=True, type=Path, help='Directory to store results.p and other outputs')
parser.add_argument('--crop', default='corn', choices=['corn', 'soy'], type=str)
parser.add_argument('--irr', default='rf', choices=['rf', 'irr'], type=str)
parser.add_argument('--features', default='default', choices=['default', 'extreme'], type=str)
parser.add_argument('--models', nargs='+', default=None, required=True)

args = parser.parse_args()

data_dir = args.datadir
input_dir = args.inputdir
out_dir = args.out
crop = args.crop
irr = args.irr
feature_set = args.features
models = args.models

if feature_set == 'default':
    features = set_default
    example_feat = 'pr_sum_gs'
elif feature_set == 'extreme':
    features = set_extreme
    example_feat = '*r10_sum_gs'
else:
    raise ValueError('Invalid feature set')

np.random.seed(42)
analysis = Analysis(features=features, out_dir=out_dir)

results = {}
start_time = time.time()
for model in models:

    print(f'Loading data for {model}')
    data = analysis.load_data(
        file_climate=data_dir / f'climate_{model}_{crop}_{irr}.h5', 
        file_yield=data_dir / f'yield_{model}_{crop}_{irr}.h5',
        file_kg=input_dir / 'Beck_KG_V1_present_0p5.tif',
        file_spam=input_dir / f'spam2010V2r0_global_H_{crop.upper()}_R_30mn.tif',
        file_soil=input_dir / 'HWSD_soil_data_on_cropland_v2.3.nc',
        file_site=input_dir / 'CLIMATEID_SLP_GGCMI_LATLON.txt',   
    )

    print(f'Running analysis for "{model}"')
    results[model] = analysis.run(model, data, calc_shap=True, calc_shap_ia=True, calc_corr=True)

    # Store (intermediate) result
    with open(out_dir / 'results.p', 'wb') as f:
        pickle.dump(results, f)

print(f'done ({(time.time() - start_time):.2f}s)')

# Save summary statistics of analysis
flat = {}
for model_name, model_res in results.items():
    for kg_name, kg_res in model_res.items():
        flat[(model_name, kg_name)] = kg_res['stats']
pd.DataFrame(flat).to_csv(out_dir / 'stats.csv')
