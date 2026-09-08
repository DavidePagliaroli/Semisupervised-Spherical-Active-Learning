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
    """Carica e normalizza il dataset Breast Cancer."""
    data = load_breast_cancer()
    X_full = data.data
    y_full = data.target  
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    
    return X_scaled, y_full

def prepara_heart():
    """Carica, imputa e codifica il dataset Heart Disease usando indici posizionali."""
    data = fetch_openml(name='heart', version=1, as_frame=False, parser='auto')
    
    # Lavoriamo direttamente con la matrice NumPy pura
    X_raw = data.data
    if hasattr(X_raw, 'toarray'):
        X_raw = X_raw.toarray()
        
    y_full = LabelEncoder().fit_transform(data.target)
    
    # Il dataset Heart Statlog (270 istanze) ha esattamente 13 feature in posizioni fisse.
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
    
    # 1. Pipeline di trasformazione (OneHotEncoding + prima standardizzazione)
    X_processed = preprocessor.fit_transform(X_raw)
    
    # 2. Centratura globale dello spazio
    X_processed = StandardScaler().fit_transform(X_processed)

    # 3. PROIEZIONE SFERICA: Normalizzazione L2
    X_processed = Normalizer(norm='l2').fit_transform(X_processed)
    
    return X_processed, y_full


def prepara_ionosphere():
    """Carica, pulisce e mappa correttamente il dataset Ionosphere per modelli Sferici."""
    data = fetch_openml(name='ionosphere', version=1, as_frame=False, parser='auto')
    
    # 1. Imputazione dei valori mancanti
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_full = imputer.fit_transform(data.data)
    
    # 2. RIMOZIONE FEATURE A VARIANZA ZERO 
    selector = VarianceThreshold(threshold=0.0)
    X_full = selector.fit_transform(X_full)
    
    # 3. MAPPATURA TOPOLOGICA CORRETTA
    # Nota: Ho corretto un piccolo refuso del tuo codice. Il commento diceva giustamente 
    # che 'g' doveva essere 0, ma l'istruzione originale mappava 'b' a 0. 
    y_full = np.where(data.target == 'g', 0, 1)
    
    # 4. Standardizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    # 5. PROIEZIONE SFERICA: Normalizzazione L2
    X_final = Normalizer(norm='l2').fit_transform(X_scaled)
    
    return X_final, y_full

def prepara_spambase():
    """Carica, imputa e normalizza il dataset Spambase."""
    data = fetch_openml(name='spambase', version=1, as_frame=False, parser='auto')
    
    # Imputa eventuali valori mancanti con la media della colonna
    imputer = SimpleImputer(strategy='mean')
    X_full = imputer.fit_transform(data.data)
    
    # Converte le etichette in 0 e 1
    y_full = LabelEncoder().fit_transform(data.target)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    
    return X_scaled, y_full

def prepara_pima():
    """Carica, pulisce e mappa correttamente il dataset Pima Indians Diabetes."""
    # 1. Caricamento del dataset da OpenML
    data = fetch_openml(name='diabetes', version=1, as_frame=False, parser='auto')
    
    X_raw = data.data
    if hasattr(X_raw, 'toarray'):
        X_raw = X_raw.toarray()
        
    # 2. Mappatura Topologica Corretta
    # In Pima, 'tested_positive' (Diabete) è la classe minoritaria (circa 268 casi).
    # 'tested_negative' (Sani) è la maggioritaria (circa 500 casi).
    # Li mappiamo rispettivamente a 0 (Centro della sfera) e 1 (Esterno).
    y_full = np.where(data.target == 'tested_positive', 0, 1)
    
    # 3. La trappola di Pima: Zeri come Valori Mancanti
    # Le feature 1 (Glucosio), 2 (Pressione), 3 (Spessore Pelle), 4 (Insulina) e 5 (BMI)
    # non possono fisicamente essere pari a 0 in un essere umano vivo. 
    # Sostituiamo questi zeri "falsi" con NaN affinché l'imputer li riconosca.
    X_df = pd.DataFrame(X_raw)
    colonne_con_zeri_anomali = [1, 2, 3, 4, 5]
    X_df[colonne_con_zeri_anomali] = X_df[colonne_con_zeri_anomali].replace(0, np.nan)
    
    # 4. Imputazione con la mediana
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_df)
    
    # 5. Trasformazione Geometrica (StandardScaler + Norma L2)
    # Applichiamo direttamente la normalizzazione L2 per spingere i punti 
    # sulla superficie di un'ipersfera, stabilizzando il margine del DADC.
    X_scaled = StandardScaler().fit_transform(X_imputed)
    X_final = Normalizer(norm='l2').fit_transform(X_scaled)
    
    return X_final, y_full