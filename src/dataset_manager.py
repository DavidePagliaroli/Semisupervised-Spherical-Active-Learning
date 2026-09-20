import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold


def prepara_breast_cancer():
    data = load_breast_cancer()
    X = data.data
    y = data.target  
    
    pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    return X, y, pipeline

def prepara_heart():
    "Carica Heart Disease e prepara la pipeline (imputazione + encoding + proiezione)."
    data = fetch_openml(name='heart', version=1, as_frame=False, parser='auto')
    
    X_raw = data.data
    if hasattr(X_raw, 'toarray'):
        X_raw = X_raw.toarray()
        
    y = LabelEncoder().fit_transform(data.target)
    
    numeric_features_idx = [0, 3, 4, 7, 9]
    categorical_features_idx = [1, 2, 5, 6, 8, 10, 11, 12]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features_idx),
            ('cat', categorical_transformer, categorical_features_idx)
        ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler_global', StandardScaler()),
        ('normalizer', Normalizer(norm='l2'))
    ])
    
    return X_raw, y, pipeline

def prepara_ionosphere():
    "Carica Ionosphere e prepara la pipeline per imputazione e scaling."
    data = fetch_openml(name='ionosphere', version=1, as_frame=False, parser='auto')
    
    y = np.where(data.target == 'g', 0, 1)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('selector', VarianceThreshold(threshold=0.0)),
        ('scaler', StandardScaler()),
        ('normalizer', Normalizer(norm='l2'))
    ])
    
    return data.data, y, pipeline

def prepara_spambase():
    "Carica Spambase e prepara la pipeline di pulizia."
    data = fetch_openml(name='spambase', version=1, as_frame=False, parser='auto')
    y = LabelEncoder().fit_transform(data.target)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    return data.data, y, pipeline

def prepara_pima():
    "Carica Pima, applica le correzioni strutturali e prepara la pipeline."
    data = fetch_openml(name='diabetes', version=1, as_frame=False, parser='auto')
    
    X_raw = data.data
    if hasattr(X_raw, 'toarray'):
        X_raw = X_raw.toarray()
        
    y = np.where(data.target == 'tested_positive', 0, 1)

    X_df = pd.DataFrame(X_raw)
    colonne_con_zeri_anomali = [1, 2, 3, 4, 5]
    X_df[colonne_con_zeri_anomali] = X_df[colonne_con_zeri_anomali].replace(0, np.nan)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('normalizer', Normalizer(norm='l2'))
    ])
    
    return X_df.values, y, pipeline
