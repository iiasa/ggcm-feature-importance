ggcms = [
    'ACEA', 'CROVER', 'CYGMA1p74', 'DSSAT-Pythia', 'EPIC-IIASA', 'ISAM', 'LDNDC', 
    'LPJ-GUESS', 'LPJmL', 'pDSSAT', 'PEPIC', 'PROMET', 'SIMPLACE-LINTUL5']

set_default = [
    'pr_sum_gs', 'rsds_sum_gs', 'tasmax_av_gs', 'tasmin_av_gs', 
    'awc', 'sand', 'silt', 'oc',    
]

set_extreme = [
    '*hdd_sum_gs', '*kdd_sum_gs', 
    '*frt_sum_gs', '*ice_sum_gs', 
    '*r10_sum_gs', '*cwd_sum_gs', '*cdd_sum_gs', '*wet_sum_gs', 
    'awc', 'sand', 'silt', 'oc', 
]

data_shifts = {
    'CYGMA1p74': 1,
    'PEPIC': -1,
    'PROMET': -1,
    'SIMPLACE-LINTUL5': -1,
}

kg_desc = {
    'A': 'tropical',
    'B': 'arid',
    'C': 'temperate',
    'D': 'cold',
    'E': 'polar'
}

scen_desc = {
    ('corn', 'rf'): 'maize, rainfed',
    ('corn', 'irr'): 'maize, irrigated',
    ('soy', 'rf'): 'soy, rainfed',
    ('soy', 'irr'): 'soy, irrigated',
}

def format_model_name(name: str) -> str:
    if name == 'IIZUMI':
        return 'GDHY'
    else:
        return name