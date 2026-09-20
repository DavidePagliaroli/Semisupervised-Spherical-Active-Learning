import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import time as tm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from dataset_manager import prepara_spambase


def main():
    print("="*60)
    print(" ADDESTRAMENTO E VALUTAZIONE SVM SU SPAMBASE (SPLIT 70-30)")
    print("="*60)

    # ---------------------------------------------------------
    # 1. CARICAMENTO DATI (grezzi + pipeline non fittata)
    # ---------------------------------------------------------
    X_raw, y_full, pipeline = prepara_spambase()
    X_raw = np.asarray(X_raw)
    y_full = np.asarray(y_full)

    valori_unici, conteggi = np.unique(y_full, return_counts=True)
    classe_minoritaria_orig = valori_unici[np.argmin(conteggi)]
    classe_maggioritaria_orig = valori_unici[np.argmax(conteggi)]

    print(f" [Topologia] Classe Minoritaria '{classe_minoritaria_orig}' -> 0")
    print(f" [Topologia] Classe Maggioritaria '{classe_maggioritaria_orig}' -> 1")

    y_full = np.where(y_full == classe_minoritaria_orig, 0, 1)

    # ---------------------------------------------------------
    # 2. SPLIT 70-30 SUI DATI GREZZI
    # ---------------------------------------------------------
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_full, test_size=0.30, random_state=42, stratify=y_full
    )

    print(" [Pre-processing] Addestramento scaler/imputer SOLO sul Training Set...")
    X_train = pipeline.fit_transform(X_train_raw)
    X_test = pipeline.transform(X_test_raw)

    print(f" [Dataset] Training Set: {X_train.shape[0]} campioni")
    print(f" [Dataset] Test Set: {X_test.shape[0]} campioni")

    # ---------------------------------------------------------
    # 3. ADDESTRAMENTO SVM
    # ---------------------------------------------------------
    svm_params = dict(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        random_state=42
    )

    print("\n" + "-"*60)
    print(" ADDESTRAMENTO SVM")
    print("-"*60)

    tempo_inizio = tm.time()
    modello = SVC(**svm_params)
    modello.fit(X_train, y_train)
    tempo_addestramento = tm.time() - tempo_inizio

    print(f" Tempo di addestramento: {tempo_addestramento:.2f} secondi")

    # ---------------------------------------------------------
    # 4. VALUTAZIONE SUL TEST SET
    # ---------------------------------------------------------
    y_pred = modello.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print("\n" + "="*60)
    print(" RISULTATI FINALI (TEST SET)")
    print("="*60)
    print(f" Accuratezza (Accuracy):  {acc*100:.2f}%")
    print(f" Precision (Macro):       {prec*100:.2f}%")
    print(f" Recall (Macro):          {rec*100:.2f}%")
    print(f" F1-Score (Macro):        {f1*100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
