"""
Module for plotting overview of GGCM importances.
"""

from pathlib import Path
import pickle
from config import set_default, scen_desc
import argparse

from plots import plot_importance


parser = argparse.ArgumentParser()
parser.add_argument('--results', default=None, required=True, type=Path, help='Path to results.p')
parser.add_argument('--out', default=None, required=True, type=Path, help='Output path; use suffix .svg or .png for desired image format')
parser.add_argument('--crop', default='corn', choices=['corn', 'soy'], type=str)
parser.add_argument('--irr', default='rf', choices=['rf', 'irr'], type=str)
args = parser.parse_args()

results_path = args.results
out_path = args.out
crop = args.crop
irr = args.irr

with open(results_path, 'rb') as f:
    results = pickle.load(f)

fig = plot_importance(data=results, features=set_default, regions='ABCD', scen_desc=scen_desc[(crop, irr)])
fig.savefig(out_path)

print('done')