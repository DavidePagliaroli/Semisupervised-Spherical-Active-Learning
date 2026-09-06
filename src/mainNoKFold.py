import warnings
# Ignora avvisi intrusivi del solutore
warnings.filterwarnings("ignore", message=".*SCS returned 2.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*qpsolvers.*")

import numpy as np
import time as tm
from sklearn.model_selection import train_test_split

# Importa le classi del tuo modello
from DADC import DADC
from ProblemaSferico import ProblemaSferico, seleziona_con_zaino, calcola_accuratezza

# Importa il tuo gestore dei dataset
from dataset_manager import (
    prepara_breast_cancer, 
    prepara_heart, 
    prepara_ionosphere, 
    prepara_spambase
)

def main():
    print("="*60)
    print(" AVVIO ADDESTRAMENTO ACTIVE LEARNING (SPLIT 70-30)")
    print("="*60)

    # ---------------------------------------------------------
    # 1. SELEZIONE E PREPARAZIONE DATI
    # ---------------------------------------------------------
    # Scegli il dataset decommentando la riga desiderata:
    X_full, y_full = prepara_heart()
    #X_full, y_full = prepara_heart()
    #X_full, y_full = prepara_ionosphere()
    #X_full, y_full = prepara_spambase()
    
    # --- ASSEGNAZIONE TOPOLOGICA DINAMICA ---
    # Contiamo quante volte appare ogni etichetta
    valori_unici, conteggi = np.unique(y_full, return_counts=True)
    classe_minoritaria_orig = valori_unici[np.argmin(conteggi)]
    classe_maggioritaria_orig = valori_unici[np.argmax(conteggi)]
    
    print(f" [Topologia] Classe Minoritaria '{classe_minoritaria_orig}' -> mappata a 0 (Set A, DENTRO)")
    print(f" [Topologia] Classe Maggioritaria '{classe_maggioritaria_orig}' -> mappata a 1 (Set B, FUORI)")
    
    # Trasformiamo rigorosamente in 0 (Minoritaria) e 1 (Maggioritaria)
    y_full = np.where(y_full == classe_minoritaria_orig, 0, 1)
    # ----------------------------------------
    
    # Split 70% Training - 30% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.30, random_state=42, stratify=y_full
    )

    print(f" [Dataset] Training Set: {X_train.shape[0]} campioni")
    print(f" [Dataset] Test Set: {X_test.shape[0]} campioni")

    
    # ---------------------------------------------------------
    # 2. INIZIALIZZAZIONE ACTIVE LEARNING
    # ---------------------------------------------------------
    N_INIT = 20  # Punti etichettati iniziali per classe
    
    # Estrazione indici iniziali per classe 0 (A) e classe 1 (B)
    idx_class_0 = np.where(y_train == 0)[0][:N_INIT]
    idx_class_1 = np.where(y_train == 1)[0][:N_INIT]
    
    A = X_train[idx_class_0]
    B = X_train[idx_class_1]
    
    # Creazione del bacino Non Etichettato (X_unlabeled)
    idx_iniziali_usati = np.concatenate([idx_class_0, idx_class_1])
    X_unlabeled = np.delete(X_train, idx_iniziali_usati, axis=0)
    Oracolo_Truth = np.delete(y_train, idx_iniziali_usati)

    # Parametri Knapsack k-KP
    richieste = 0
    MAX_ROUND = 10
    BUDGET_GLOBALE = 100
    c_base = 1.0
    W = 4.0
    Q = 10  # Limite massimo di query per iterazione
    
    budget_consumato = 0.0
    iterazione = 1
    storico_accuratezze = []

    # ---------------------------------------------------------
    # 3. CICLO DI ACTIVE LEARNING
    # ---------------------------------------------------------
    tempo_inizio = tm.time()
    
    # --- VARIABILE WARM START ---
    memoria_sfera = None

    while budget_consumato < BUDGET_GLOBALE and iterazione <= MAX_ROUND:
        budget_rimanente = BUDGET_GLOBALE - budget_consumato

        # Stop se non ci sono dati o se il budget residuo non copre il costo base
        if len(X_unlabeled) == 0 or budget_rimanente < c_base:
            break
            
        print(f"\n[Iter {iterazione}] Budget Consumato: {budget_consumato:.2f}/{BUDGET_GLOBALE:.2f}")
        print(f"  Stato: |A|={len(A)}, |B|={len(B)}, |X|={len(X_unlabeled)}")

        # --- A. ADDESTRAMENTO (Con Warm Start) ---
        prob = ProblemaSferico(A, B, X_unlabeled, C1=0.05, C2=0.005, start_v=memoria_sfera) 
        DADC(prob)
        
        # Salviamo la sfera trovata per usarla come partenza al prossimo giro
        sfera_ottimizzata = prob.Xk 
        memoria_sfera = np.copy(sfera_ottimizzata)
        
        # --- B. VALUTAZIONE SUL TEST SET ---
        acc = calcola_accuratezza(sfera_ottimizzata, X_test, y_test, 0, 1)
        storico_accuratezze.append(acc)
        print(f"  [Metriche] Accuratezza Test Set: {acc*100:.2f}%")

        # --- C. SELEZIONE CON ZAINO (Branch & Bound k-KP) ---
        indici_scelti, valori_vj, costo_speso = seleziona_con_zaino(
            sfera_v=sfera_ottimizzata, 
            X_unlabeled=X_unlabeled, 
            budget_attuale=budget_rimanente, 
            C2=prob.C2, 
            c_base=c_base, 
            W=W, 
            max_elementi=Q
        )

        # --- D. EARLY STOPPING ---
        if len(indici_scelti) == 0:
            print("  [STOP ANTICIPATO] Nessun punto informativo selezionabile. Modello sicuro o budget insufficiente.")
            break

        # --- E. AGGIORNAMENTO ORACOLO ---
        for idx in sorted(indici_scelti, reverse=True):
            punto = X_unlabeled[idx]
            etichetta_reale = Oracolo_Truth[idx]

            if etichetta_reale == 0:
                A = np.vstack([A, punto])
            else:
                B = np.vstack([B, punto])
            richieste+=1
            X_unlabeled = np.delete(X_unlabeled, idx, axis=0)
            Oracolo_Truth = np.delete(Oracolo_Truth, idx)

        budget_consumato += costo_speso
        iterazione += 1

    # ---------------------------------------------------------
    # 4. RISULTATI FINALI
    # ---------------------------------------------------------
    tempo_totale = tm.time() - tempo_inizio
    
    print("\n" + "="*60)
    print(" ADDESTRAMENTO COMPLETATO!")
    print(f" Tempo di esecuzione: {tempo_totale:.2f} secondi")
    print(f" Iterazioni effettuate: {iterazione -1}")
    print(f" Budget totale speso: {budget_consumato:.2f}")
    print(f" Richieste effettuate all'oracolo: {richieste}")
    print(f" Accuratezza Massima Raggiunta: {max(storico_accuratezze)*100:.2f}%")
    print(" Evoluzione Accuratezza:")
    for i, acc in enumerate(storico_accuratezze):
        print(f"   Iter {i+1}: {acc*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()