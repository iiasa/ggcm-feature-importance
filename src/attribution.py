from typing import List
import warnings
import numpy as np
import pandas as pd
from sklearn import metrics
from xgboost import XGBClassifier
import time
from PIL import Image
from scipy.stats import qmc

import math
import pickle
import xarray as xr
import shap

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import brier_score_loss
from sklearn.metrics import average_precision_score


class Analysis:

    kg_regions = [
        None, 'Af', 'Am', 'Aw', 'BWh', 'BWk', 'BSh', 'BSk', 'Csa', 'Csb', 'Csc', 'Cwa', 'Cwb', 'Cwc', 'Cfa', 'Cfb', 'Cfc', 
        'Dsa', 'Dsb', 'Dsc', 'Dsd', 'Dwa', 'Dwb', 'Dwc', 'Dwd', 'Dfa', 'Dfb', 'Dfc', 'Dfd', 'ET', 'EF'
    ]

    def __init__(self, features: List[str], out_dir: Path):
        self.out_dir = out_dir
        self.features, self.is_day_feature = zip(*[(x[1:], True) if x.startswith('*') else (x, False) for x in features])
        self.features = list(self.features)
    
    def load_data(
        self, file_climate: Path, file_yield: Path, file_kg: Path, file_spam: Path, file_soil: Path, 
        file_site: Path, shift_lon: bool = False, features_to_normalize: List[str] = None) -> pd.DataFrame:

        # Load climate data
        data =  pd.read_hdf(file_climate)

        # Join yields
        data_yield = pd.read_hdf(file_yield)
        data_yield = data_yield[data_yield['yield'] > 0]

        if shift_lon:
            data_yield.index = data_yield.index.set_levels(((data_yield.index.levels[1] - 180) % 360) - 180, level=1)

        data = data.join(data_yield, how='inner')

        # Compute classification label; negative yield anomalies -15% of the expected value are considered anomalies.
        data['is_anomaly'] = data['rel_yield'] < -0.15

        # Normalize day features by growing season length
        if features_to_normalize:
            data[features_to_normalize] = data[features_to_normalize] / data[['gs_length']].values

        # Pixel-level quantities (pixel mean yield, Köppen-Geiger region)
        grouped = data.groupby(['lat', 'lon']).mean()
        y = self._lat_to_idx(grouped.index.get_level_values('lat').values).astype(int)
        x = self._lon_to_idx(grouped.index.get_level_values('lon').values).astype(int)

        # Read Köppen-Geiger classification by Beck et al.
        with Image.open(file_kg) as img:
            grouped['kg'] = np.array(img)[y, x]

        # Read MAPSPAM harvested area
        with Image.open(file_spam) as img:
            grouped['h'] = np.array(img)[y, x]

        # Read soil data
        features_soil = []
        with xr.open_dataset(file_soil, decode_times=False) as soil:
            soil_df = soil.to_dataframe()
            features_soil = [x for x in self.features if x in soil_df.columns]
            grouped = grouped.join(soil_df[features_soil])
        
        if any(np.isnan(grouped).sum(axis=0) > grouped.shape[0] * 0.1):
            warnings.warn('More than 10% nans')

        # Read site data
        slope = pd.read_csv(file_site, index_col=[0, 1])
        slope.columns = ['slp']
        slope.index.names = ['lat', 'lon']
        grouped = grouped.join(slope)
        features_soil.append('slp')

        # Merge data sources
        data = data.join(grouped[['yield', 'kg', 'h'] + features_soil], rsuffix='_mean', how='left')
        data = data[data['h'] > 0]
        data = data[data['kg'] > 0]
        
        data.dropna(inplace=True)

        return data

    def run(
        self, model_name: str, data: pd.DataFrame, calc_shap: bool, calc_shap_ia: bool, calc_corr: bool, 
        do_cv: bool = False, sample_size: int = 1000, regions: str = 'ABCDE') -> dict:
        
        results = {}
        h_total = data.groupby(level=['lat', 'lon']).first()['h'].sum()
        for kg in regions:
            print(f'Processing region {kg}')

            kg_indices = [self.kg_regions.index(x) for x in self.kg_regions if x is not None and x.startswith(kg)]
            kg_data = data[data['kg'].isin(kg_indices)]
            data_train = kg_data
            data_shap = kg_data[self.features + ['is_anomaly']].sample(min(sample_size, kg_data.shape[0]))

            fit_fn = self._fit_xgb_cv if do_cv else self._fit_xgb
            model, conf, auroc, briar, auprc, n_train = fit_fn(data_train, model_name, kg, do_cache=False)
            if model is None:
                print(f'No data for {kg}/{model_name}')
                continue
            elif conf.shape == (1, 1):
                print(f'No anomalies in {kg}/{model_name}')
                continue

            # Collect KG/GGCM data statistics
            kg_area = kg_data['h'].sum()
            kg_area = kg_data.groupby(level=['lat', 'lon']).first()['h'].sum()
            results[kg] = {}
            results[kg]['stats'] = self._stats(conf, kg_area, kg_area / h_total, kg_data.shape[0], np.sum(kg_data['is_anomaly']))
            results[kg]['stats']['auroc'] = auroc
            results[kg]['stats']['briar'] = briar
            results[kg]['stats']['auprc'] = auprc
            results[kg]['stats']['n_train'] = n_train
            print(f"Recall: {results[kg]['stats']['recall']:.2f}, Precision: {results[kg]['stats']['precision']:.2f}, AUROC: {results[kg]['stats']['auroc']:.2f}, Briar: {results[kg]['stats']['briar']:.2f}")

            if calc_shap:
                print('Calculating SHAP values... ', end='')
                start_time = time.time()
                explainer = shap.TreeExplainer(model, data=kg_data[self.features], feature_perturbation='interventional', model_output='probability')
                results[kg]['shap'] = explainer(data_shap[self.features])
                results[kg]['shap_expected'] = explainer.expected_value
                results[kg]['shap_coords'] = data_shap.index
                results[kg]['p_ano'] = model.predict_proba(data_shap[self.features])[:, 1]
                print(f'{(time.time() - start_time):.2f}s')
                
            if calc_shap_ia:
                print('Calculating SHAP interactions... ', end='')
                start_time = time.time()
                # 2024-12-17: interventional perturbation for interactions currently not supported in SHAP. Use log-odds model and record 
                # expected value to transform shaps to probabilities by doing something like this:
                # ias_expected_p = np.exp(explainer2.expected_value) / (1 + np.exp(explainer2.expected_value))
                # ias_p = np.exp(ias_abs) / (1 + np.exp(ias_abs)) - ias_expected_p
                explainer = shap.TreeExplainer(model)
                results[kg]['shap_ia'] = explainer.shap_interaction_values(data_shap[self.features])
                results[kg]['shap_ia_expected'] = explainer.expected_value
                print(f'{(time.time() - start_time):.2f}s')
            if calc_corr:
                print('Calculating correlations... ', end='')
                start_time = time.time()
                results[kg]['corr'] = kg_data[self.features].corr()
                results[kg]['corr_train'] = data_train[self.features].corr()
                print(f'{(time.time() - start_time):.2f}s')

        return results

    def _stats(self, conf: np.ndarray, area: float, area_frac: float, n_total: int, n_anomaly: int):
        tn, fp, fn, tp = conf.flatten()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        anomaly_frac = n_anomaly / n_total
        return {
            'area': area,
            'area_frac': area_frac,
            'anomaly_frac': anomaly_frac,
            'n_anomaly': n_anomaly,
            'n_total': n_total,
            'misc_rate': (conf[1, 0] + conf[0, 1]) / conf.sum(), 
            'recall': recall,
            'precision': precision,
            'f1': f1,
        }

    def _sample_olhs(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        (Semi-)orthogonal latin hypercube sampling
        """
        features_var = [k for k, v in data[self.features].var().items() if v > 0]

        qt = QuantileTransformer(output_distribution='uniform', n_quantiles=10, random_state=42)
        qt.fit(data[features_var])

        num_rows = min(10000, data.shape[0])
        num_features = len(features_var)
        num_samples = self._nearest_prime_square(num_rows)

        sampler = qmc.LatinHypercube(d=num_features, strength=2)
        lhs_samples = pd.DataFrame(sampler.random(n=num_samples), columns=features_var)
        data_sampled = qt.inverse_transform(lhs_samples)
        df_sampled = pd.DataFrame(data_sampled, columns=features_var)

        nbrs = NearestNeighbors(n_neighbors=1).fit(data[features_var])
        distances, indices = nbrs.kneighbors(df_sampled)
        nns = data.iloc[indices.flatten()].astype({'is_anomaly': bool})

        return nns
            
    def _lat_to_idx(self, lat):
        return (-lat + 89.75) * 2

    def _lon_to_idx(self, lon):
        return (lon + 179.75) * 2
    
    def _fit_xgb(self, data_train, model_name: str, kg_name: str, data_test = None, do_cache: bool = False):
        if data_train.shape[0] == 0:
            return None, None, None, None, None, None

        if data_test is None:
            x_train, x_test, y_train, y_test = train_test_split(
                data_train[self.features], data_train['is_anomaly'], stratify=data_train['is_anomaly'], shuffle=True, test_size=0.1)
        else:
            x_train = data_train[self.features]
            y_train = data_train['is_anomaly']
            x_test = data_test[self.features]
            y_test = data_test['is_anomaly']

        n_estimators = 150 if kg_name != 'E' else 100
        max_depth = 12 if model_name not in ['DSSAT-Pythia', 'pDSSAT', 'IIZUMI'] else 15

        model = XGBClassifier(
            objective='binary:logistic', 
            n_estimators=n_estimators,
            max_depth=max_depth, 
            importance_type='total_gain', 
            min_child_weight=5,
            gamma=0.3,
            tree_method='hist',
            random_state=42)

        model.fit(x_train, y_train)
        conf_matrix = metrics.confusion_matrix(y_test, model.predict(x_test))
        
        # Test set eval
        y_pred_proba = model.predict_proba(x_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        briar = brier_score_loss(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)
        
        return model, conf_matrix, roc_auc, briar, auprc, x_train.shape[0]

    def _fit_xgb_cv(self, data, model_name: str, kg_name: str, do_cache: bool = True):

        if do_cache:
            cache_dir = (self.out_dir / 'model_cache')
            cache_dir.mkdir(exist_ok=True)
            cache_path = cache_dir / f'{model_name}_{kg_name}.p'

            if cache_path.exists():
                print(f'Loading model {model_name}/{kg_name} from cache')
                with open(cache_path, 'rb') as f:
                    cache = pickle.load(f)
                return cache['model'], cache['conf_matrix'], cache['auroc'], cache['n_train']

        if data.shape[0] == 0:
            return

        x = data[list(self.features)]
        y = data['is_anomaly']
        x_train, y_train = x, y
        
        param_dist = {
            'n_estimators': [10, 50, 100, 150],
            'learning_rate': [0.05, 0.1, 0.2, 0.3],
            'max_depth': [11, 13, 15, 17],
            'min_child_weight': [1, 3, 5, 7, 10],
            'gamma': [0, 0.3, 0.5, 1.0],
        }

        model_template = XGBClassifier(
            objective='binary:logistic', 
            importance_type='total_gain', 
            tree_method='hist',
            random_state=42,
        )

        random_search = RandomizedSearchCV(
            estimator=model_template,
            param_distributions=param_dist,
            n_iter=50,
            scoring='roc_auc', 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), # 10
            verbose=1,
            n_jobs=-1,
            random_state=42
        )

        start_time = time.time()
        random_search.fit(x_train, y_train)
        print(f'Random grid search took {(time.time() - start_time):.2f}s')
        print("Best Parameters:", random_search.best_params_)

        model = random_search.best_estimator_
        conf_matrix = metrics.confusion_matrix(y_train, model.predict(x_train))
        
        if do_cache:
            with open(cache_dir / f'{model_name}_{kg_name}.p', 'wb') as f:
                pickle.dump({'model': model, 'conf_matrix': conf_matrix, 'auroc': random_search.best_score_, 'n_train': x_train.shape}, f)

        return model, conf_matrix, random_search.best_score_, x_train.shape[0]

    def _nearest_prime_square(self, n):
        if n < 4:
            return 4
        return self._next_prime_lt(int(math.sqrt(n)))**2

    def _next_prime_lt(self, n):
        if (n & 1):
            n -= 2
        else:
            n -= 1
        i, j = 0, 3
        for i in range(n, 2, -2):
            if(i % 2 == 0):
                continue
            while(j <= math.floor(math.sqrt(i)) + 1): 
                if (i % j == 0):
                    break
                j += 2
            if (j > math.floor(math.sqrt(i))):
                return i
        return 2

