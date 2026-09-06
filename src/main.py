
import time as tm
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

# Importa i tuoi moduli
from dataset_manager import prepara_breast_cancer, prepara_spambase, prepara_ionosphere, prepara_heart, prepara_pima
from ProblemaSferico import ProblemaSferico, calcola_accuratezza, seleziona_con_zaino
from DADC import DADC
import warnings
# Ignora gli avvisi di deprecazione generati dalle librerie sottostanti
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
if __name__ == "__main__":
    # 1. Caricamento del dataset intero
    X, y = prepara_heart()

    BUDGET_GLOBALE = 20
    BUDGET_QUERY = 4  #"""NON VOGLIO UN BUDGET QUERY MA UN NUMERO MASSIMO DI CAMPIONI CONSIDERATI PER ITERAZIONE"""
    # Parametri di costo eterogeneo per il Knapsack
    c_base = 1.0
    W = 4.0
    Q = 10
    # 2. Configurazione della 10-Fold Cross-Validation
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuratezze_finali_folds = []
    
    tempo_totale_inizio = tm.time()

    # 3. Ciclo sui 10 Fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
        print(f"\n{'='*70}")
        print(f" AVVIO FOLD {fold}/10")
        print(f"{'='*70}")
        
        # Dividiamo i dati per questo fold (90% training, 10% test)
        X_train_full = X[train_idx]
        y_train_full = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        # La regola della prof: nel set di addestramento, 30% etichettato iniziale, 70% non etichettato
        X_initial, X_unlabeled, y_initial, Oracolo_Truth = train_test_split(
            X_train_full, y_train_full, 
            test_size=0.70, # 70% va nel pool non etichettato
            stratify=y_train_full, 
            random_state=42
        )
        
        #l'algoritmo funziona meglio se la classe con la cardinalità minore si trova dentro la sfera
        
        # 1. Analizziamo le proporzioni delle classi nel set iniziale
        classi, conteggi = np.unique(y_initial, return_counts=True)
        
        # 2. Identifichiamo dinamicamente chi è la minoranza e chi la maggioranza
        classe_minoritaria = classi[np.argmin(conteggi)]
        classe_maggioritaria = classi[np.argmax(conteggi)]
        
        print(f"  [Topologia] Sfera centrata sulla classe {classe_minoritaria} (minoritaria).")

        # 3. Assegniamo in modo intelligente A (dentro la sfera) e B (fuori)
        A = X_initial[y_initial == classe_minoritaria].copy()
        B = X_initial[y_initial == classe_maggioritaria].copy()
        
        etichette_consumate = 0
        iterazione = 1
        miglior_acc_fold = 0.0

# Variabile per conservare la memoria della sfera tra un'iterazione e l'altra
        memoria_sfera = None

        # --- CICLO DI ACTIVE LEARNING ---
        while budget_consumato < BUDGET_GLOBALE:
            budget_rimanente = BUDGET_GLOBALE - budget_consumato
            richiesta_attuale = min(BUDGET_QUERY, budget_rimanente)

             # Stop se non ci sono dati o se il budget residuo non basta nemmeno per il punto più economico
            if len(X_unlabeled) == 0 or richiesta_attuale < c_base:
                break
            
            print(f"\n  [Fold {fold} - Iter {iterazione}] Budget Speso: {budget_consumato:.2f}/{BUDGET_GLOBALE:.2f}")
            print(f"  Stato: |A|={len(A)}, |B|={len(B)}, |X|={len(X_unlabeled)}")

            # Addestramento con WARM START
            prob = ProblemaSferico(A, B, X_unlabeled, C1=0.01, C2=0.01, start_v=memoria_sfera)
            DADC(prob)
            
            # Salviamo il risultato per il prossimo ciclo
            sfera_ottimizzata = prob.Xk 
            memoria_sfera = sfera_ottimizzata 
            
            # Valutazione
            acc = calcola_accuratezza(sfera_ottimizzata, X_test, y_test, classe_minoritaria, classe_maggioritaria)            
            if acc > miglior_acc_fold:
                miglior_acc_fold = acc
                
            print(f"  [Metriche] Accuratezza Test Set: {acc*100:.2f}%")

            # Selezione con Zaino (Estraiamo anche il costo_speso ed esplicitiamo c_base e W)
            indici_scelti, valori_vj, costo_speso = seleziona_con_zaino(sfera_ottimizzata, X_unlabeled, richiesta_attuale, prob.C2, c_base, W, Q)

            # --- CONDIZIONE DI EARLY STOPPING ---
            if len(indici_scelti) == 0:
                print("  [STOP ANTICIPATO] Nessun punto informativo selezionabile. Modello sicuro o budget insufficiente.")
                break

            # Aggiornamento Oracolo 
            for idx in sorted(indici_scelti, reverse=True):
                punto = X_unlabeled[idx]
                etichetta_reale = Oracolo_Truth[idx]

                if etichetta_reale == 0:
                    A = np.vstack([A, punto])
                else:
                    B = np.vstack([B, punto])

                X_unlabeled = np.delete(X_unlabeled, idx, axis=0)
                Oracolo_Truth = np.delete(Oracolo_Truth, idx)

            # Consumiamo il budget usando il costo esatto calcolato dal Branch & Bound
            budget_consumato += costo_speso
            iterazione += 1
            
        # Fine del Fold: salviamo il miglior risultato ottenuto
        accuratezze_finali_folds.append(miglior_acc_fold)
        print(f"  -> Miglior accuratezza nel Fold {fold}: {miglior_acc_fold*100:.2f}%")

    # --- RISULTATI GLOBALI CROSS VALIDATION ---
    tempo_totale = tm.time() - tempo_totale_inizio
    media_acc = np.mean(accuratezze_finali_folds)
    dev_std = np.std(accuratezze_finali_folds)
    
    print(f"\n{'*'*70}")
    print(" 10-FOLD CROSS VALIDATION COMPLETATA!")
    print(f" Tempo totale di esecuzione: {tempo_totale:.2f} secondi")
    print(f" Accuratezza Media: {media_acc*100:.2f}% ± {dev_std*100:.2f}%")
    print(f" Storico Folds: {[round(a*100, 1) for a in accuratezze_finali_folds]}")
    print(f"{'*'*70}")