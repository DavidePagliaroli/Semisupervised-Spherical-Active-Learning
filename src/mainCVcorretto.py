import warnings
# Ignora avvisi intrusivi del solutore
warnings.filterwarnings("ignore", message=".*SCS returned 2.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*qpsolvers.*")

import numpy as np
import time as tm
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score 

# Importa le classi del tuo modello
from DADC import DADC
from ProblemaSferico import ProblemaSferico, seleziona_con_zaino, calcola_accuratezza

from dataset_manager import (
    prepara_breast_cancer, 
    prepara_heart, 
    prepara_ionosphere, 
    prepara_spambase,
    prepara_pima
)

def main():
    print("="*70)
    print(" AVVIO ADDESTRAMENTO ACTIVE LEARNING (10-FOLD CROSS VALIDATION)")
    print("="*70)

    # ---------------------------------------------------------
    # 1. SELEZIONE E PREPARAZIONE DATI
    # ---------------------------------------------------------
    #X_raw, y_full, pipeline_template = prepara_breast_cancer()
    #X_raw, y_full, pipeline_template = prepara_heart()
    #X_raw, y_full, pipeline_template = prepara_ionosphere()
    X_raw, y_full, pipeline_template = prepara_spambase()
    #X_raw, y_full, pipeline_template = prepara_pima()

    X_raw = np.asarray(X_raw)
    y_full = np.asarray(y_full)

    valori_unici, conteggi = np.unique(y_full, return_counts=True)
    classe_minoritaria_orig = valori_unici[np.argmin(conteggi)]
    classe_maggioritaria_orig = valori_unici[np.argmax(conteggi)]
    
    print(f" [Topologia] Classe Minoritaria '{classe_minoritaria_orig}' -> 0 (DENTRO)")
    print(f" [Topologia] Classe Maggioritaria '{classe_maggioritaria_orig}' -> 1 (FUORI)")
    
    y_full = np.where(y_full == classe_minoritaria_orig, 0, 1)

    # ---------------------------------------------------------
    # 2. PARAMETRI ACTIVE LEARNING E MODELLO
    # ---------------------------------------------------------

    C1_val = 0.01  
    C2_val = 0.01   
    PERCENTUALE_INIZIALE = 0.072 
    
    MAX_ROUND = 15
    BUDGET_GLOBALE = 700.0
    BUDGET_PER_ITERAZIONE = 80.0
    c_base = 1.0
    W = 4.0
    MAX_SAMPLE = 30
    
    K_FOLDS = 10
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    risultati_cv = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'richieste_oracolo': [],
        'tempi': []
    }

    tempo_inizio_cv = tm.time()

    # ---------------------------------------------------------
    # 3. CICLO SUI FOLD DELLA CROSS-VALIDATION
    # ---------------------------------------------------------
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_raw, y_full), 1):
        print("\n" + "*"*70)
        print(f" ESECUZIONE FOLD {fold}/{K_FOLDS}")
        print("*"*70)
        
        tempo_inizio_fold = tm.time()
        
        X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y_full[train_idx], y_full[test_idx]

        pipeline = clone(pipeline_template)
        X_train = pipeline.fit_transform(X_train_raw)
        X_test = pipeline.transform(X_test_raw)
        
        X_seed, X_unlabeled, y_seed, Oracolo_Truth = train_test_split(
            X_train, y_train, 
            train_size=PERCENTUALE_INIZIALE, 
            random_state=42, 
            stratify=y_train
        )
        
        A = X_seed[y_seed == 0]
        B = X_seed[y_seed == 1]
        
        if len(A) == 0 or len(B) == 0:
            raise ValueError(f"Fold {fold}: Il seed non contiene rappresentanti per entrambe le classi.")

        budget_consumato = 0.0
        iterazione = 1
        richieste = 0
        memoria_sfera = None
        totale_etichettati = len(A) + len(B)

        # --- A. MODELLO INIZIALE (Cold Start) ---
        prob = ProblemaSferico(A, B, X_unlabeled, C1=C1_val, C2=C2_val) 
        DADC(prob)
        sfera_ottimizzata = prob.Xk 
        memoria_sfera = np.copy(sfera_ottimizzata)
        
        acc_iniziale = calcola_accuratezza(sfera_ottimizzata, X_test, y_test, 0, 1)
        print(f"  [Cold Start] Accuratezza Iniziale: {acc_iniziale*100:.2f}% (su {totale_etichettati} campioni)")

        # --- B. CICLO ACTIVE LEARNING ---
        while budget_consumato < BUDGET_GLOBALE and iterazione <= MAX_ROUND:
            budget_rimanente = BUDGET_GLOBALE - budget_consumato

            if len(X_unlabeled) == 0 or budget_rimanente < c_base:
                break

            budget_zaino = min(BUDGET_PER_ITERAZIONE, budget_rimanente)
            indici_scelti, valori_vj, costo_speso = seleziona_con_zaino(
                sfera_v=sfera_ottimizzata, 
                X_unlabeled=X_unlabeled, 
                budget_attuale=budget_zaino, 
                C2=prob.C2, 
                c_base=c_base, 
                W=W, 
                max_elementi=MAX_SAMPLE
            )

            if len(indici_scelti) == 0:
                print(f"  [Iter {iterazione}] STOP ANTICIPATO (Modello sicuro o budget insufficiente).")
                break

            for idx in sorted(indici_scelti, reverse=True):
                punto = X_unlabeled[idx]
                etichetta_reale = Oracolo_Truth[idx]
                if etichetta_reale == 0:
                    A = np.vstack([A, punto])
                else:
                    B = np.vstack([B, punto])
                richieste += 1
                X_unlabeled = np.delete(X_unlabeled, idx, axis=0)
                Oracolo_Truth = np.delete(Oracolo_Truth, idx)

            budget_consumato += costo_speso
            totale_etichettati = len(A) + len(B)
            
            prob = ProblemaSferico(A, B, X_unlabeled, C1=C1_val, C2=C2_val, start_v=memoria_sfera) 
            DADC(prob)
            sfera_ottimizzata = prob.Xk 
            memoria_sfera = np.copy(sfera_ottimizzata)
            
            acc = calcola_accuratezza(sfera_ottimizzata, X_test, y_test, 0, 1)
            print(f"  [Iter {iterazione}] Acc: {acc*100:.2f}% | Budget: {budget_consumato:.1f}/{BUDGET_GLOBALE:.1f} | Tot etichettati: {totale_etichettati}")
            
            iterazione += 1

        # --- C. CALCOLO METRICHE FINALI DEL FOLD ---
        tempo_fold = tm.time() - tempo_inizio_fold
        
        x0_final = np.asarray(sfera_ottimizzata[:-2], dtype=float)
        z_final = float(sfera_ottimizzata[-2])
        dist_sq_test = np.sum((X_test - x0_final)**2, axis=1)
        y_pred = np.where(dist_sq_test <= z_final, 0, 1)
        
        acc_finale = calcola_accuratezza(sfera_ottimizzata, X_test, y_test, 0, 1)
        prec_finale = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_finale = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_finale = f1_score(y_test, y_pred, average='macro', zero_division=0)

        risultati_cv['accuracy'].append(acc_finale)
        risultati_cv['precision'].append(prec_finale)
        risultati_cv['recall'].append(rec_finale)
        risultati_cv['f1'].append(f1_finale)
        risultati_cv['richieste_oracolo'].append(richieste)
        risultati_cv['tempi'].append(tempo_fold)
        
        print(f"  -> RISULTATI FOLD {fold}: Accuracy={acc_finale*100:.2f}%, Query={richieste}, Tempo={tempo_fold:.1f}s")

    # ---------------------------------------------------------
    # 4. RIEPILOGO GLOBALE 10-FOLD CV
    # ---------------------------------------------------------
    tempo_totale_cv = tm.time() - tempo_inizio_cv
    
    print("\n" + "="*70)
    print(" 10-FOLD CROSS VALIDATION COMPLETATA")
    print(f" Tempo Totale di Esecuzione: {tempo_totale_cv:.2f} secondi")
    print("="*70)
    print("\n[ DATI PER TABELLA LATEX - ANDAMENTO ACCURATEZZA ]")
    
    for i, acc in enumerate(risultati_cv['accuracy'], 1):
        print(f" Fold {i:<2}: {acc*100:.2f}%")
        
    media_acc = np.mean(risultati_cv['accuracy']) * 100
    std_acc = np.std(risultati_cv['accuracy']) * 100
    print("-" * 30)
    print(f" MEDIA  : {media_acc:.2f}% ± {std_acc:.2f}%")
    
    print("\n[ METRICHE GLOBALI MEDIE (MACRO) ]")
    print(f" Accuracy  : {media_acc:.2f}% ± {std_acc:.2f}%")
    print(f" Precision : {np.mean(risultati_cv['precision'])*100:.2f}%")
    print(f" Recall    : {np.mean(risultati_cv['recall'])*100:.2f}%")
    print(f" F1-Score  : {np.mean(risultati_cv['f1'])*100:.2f}%")
    print(f" Oracolo   : ~ {int(np.mean(risultati_cv['richieste_oracolo']))} query medie a fold")
    print("="*70)

if __name__ == "__main__":
    main()