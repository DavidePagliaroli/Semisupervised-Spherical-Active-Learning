import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
#from dataset_manager import prepara_heart
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def prepara_heart():
    """Carica, imputa e codifica il dataset Heart inferendo automaticamente i tipi di colonna."""
    data = fetch_openml(name='heart', version=1, as_frame=False, parser='auto')
    X_raw = data.data
    
    if hasattr(X_raw, 'toarray'):
        X_raw = X_raw.toarray()
        
    X_df = pd.DataFrame(X_raw, columns=data.feature_names)
    y_full = LabelEncoder().fit_transform(data.target)
    
    # Rilevamento automatico: se una colonna ha meno di 10 valori unici, è categorica
    categorical_features = []
    numeric_features = []
    
    for col in X_df.columns:
        if X_df[col].nunique() < 10:
            categorical_features.append(col)
        else:
            numeric_features.append(col)
            
    # Pipeline per le numeriche (imputazione mediana + standardizzazione)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline per le categoriali (imputazione moda + one-hot encoding)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    X_processed = preprocessor.fit_transform(X_df)
    
    return X_processed, y_full
# 1. Caricamento degli stessi identici dati del DADC
X, y = prepara_heart()

# 2. Setup della 10-Fold Cross Validation stratificata
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

acc_rf_list = []
acc_svc_list = []

print("Avvio test diagnostico su 10 Fold...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Modello 1: Random Forest (partizionamento ad albero, ignora la topologia sferica)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf_list.append(accuracy_score(y_test, rf.predict(X_test)))

    # Modello 2: SVC standard con Kernel RBF (distorce lo spazio per trovare separabilità non lineare)
    svc = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svc.fit(X_train, y_train)
    acc_svc_list.append(accuracy_score(y_test, svc.predict(X_test)))

print("\n--- RISULTATI BASELINE DIAGNOSTICA ---")
print(f"Random Forest (Accuratezza Media 10-Fold): {np.mean(acc_rf_list) * 100:.2f}%")
print(f"SVC Kernel RBF (Accuratezza Media 10-Fold): {np.mean(acc_svc_list) * 100:.2f}%")