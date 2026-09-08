import warnings
# Ignora avvisi intrusivi del solutore
warnings.filterwarnings("ignore", message=".*SCS returned 2.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*qpsolvers.*")

import numpy as np
import time as tm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score # <--- NUOVO IMPORT

# Importa le classi del tuo modello
from DADC import DADC
from ProblemaSferico import ProblemaSferico, seleziona_con_zaino, calcola_accuratezza

# Importa il tuo gestore dei dataset
from dataset_manager import (
    prepara_breast_cancer, 
    prepara_heart, 
    prepara_ionosphere, 
    prepara_spambase,
    prepara_pima
)

def main():
    print("="*60)
    print(" AVVIO ADDESTRAMENTO ACTIVE LEARNING (SPLIT 70-30)")
    print("="*60)

    # ---------------------------------------------------------
    # 1. SELEZIONE E PREPARAZIONE DATI
    # ---------------------------------------------------------
    # Scegli il dataset decommentando la riga desiderata:
    #X_full, y_full = prepara_breast_cancer()
    #X_full, y_full = prepara_heart()
    #X_full, y_full = prepara_ionosphere()
    X_full, y_full = prepara_spambase()
    #X_full, y_full = prepara_pima()
    
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
    # 2. INIZIALIZZAZIONE ACTIVE LEARNING (COLD START)
    # ---------------------------------------------------------
    PERCENTUALE_INIZIALE = 0.10  # 10% del training set
    
    # Suddivisione stratificata del Training Set in Seed (svelato) e Unlabeled (cieco)
    X_seed, X_unlabeled, y_seed, Oracolo_Truth = train_test_split(
        X_train, y_train, 
        train_size=PERCENTUALE_INIZIALE, 
        random_state=42, 
        stratify=y_train
    )
    
    print(f" [Cold Start] Svelato il {PERCENTUALE_INIZIALE*100:.0f}% del Training Set ({len(X_seed)} campioni stratificati).")

    # Costruzione dei set iniziali A (classe 0) e B (classe 1) sulla base del seed
    A = X_seed[y_seed == 0]
    B = X_seed[y_seed == 1]
    
    # Verifica di sicurezza (requisito matematico per non avere singolarità)
    if len(A) == 0 or len(B) == 0:
        raise ValueError("Il campionamento iniziale non contiene almeno un rappresentante per ciascuna classe. Aumentare la percentuale o controllare il dataset.")

    # Parametri Knapsack k-KP
    richieste = 0
    MAX_ROUND = 10
    BUDGET_GLOBALE = 150
    BUDGET_PER_ITERAZIONE = 15.0
    c_base = 1.0
    W = 4.0
    MAX_SAMPLE = 30  # Limite massimo di query per iterazione
    
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
        prob = ProblemaSferico(A, B, X_unlabeled, C1=0.01, C2=0.01, start_v=memoria_sfera) 
        DADC(prob)
        
        # Salviamo la sfera trovata per usarla come partenza al prossimo giro
        sfera_ottimizzata = prob.Xk 
        memoria_sfera = np.copy(sfera_ottimizzata)
        
        # --- B. VALUTAZIONE SUL TEST SET ---
        acc = calcola_accuratezza(sfera_ottimizzata, X_test, y_test, 0, 1)
        storico_accuratezze.append(acc)
        print(f"  [Metriche] Accuratezza Test Set: {acc*100:.2f}%")

        # --- C. SELEZIONE CON ZAINO (Branch & Bound k-KP) ---
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
            richieste += 1
            X_unlabeled = np.delete(X_unlabeled, idx, axis=0)
            Oracolo_Truth = np.delete(Oracolo_Truth, idx)

        budget_consumato += costo_speso
        iterazione += 1

    # ---------------------------------------------------------
    # 4. RISULTATI FINALI E METRICHE GLOBALI
    # ---------------------------------------------------------
    tempo_totale = tm.time() - tempo_inizio
    
    # --- CALCOLO PREDIZIONI SUL TEST SET (Sfera Ottimizzata) ---
    # Ricostruiamo la geometria finale
    x0_final = np.asarray(sfera_ottimizzata[:-2], dtype=float)
    z_final = float(sfera_ottimizzata[-2])
    
    # Distanza al quadrato dal centro
    dist_sq_test = np.sum((X_test - x0_final)**2, axis=1)
    # Se d <= z, classe 0 (DENTRO). Altrimenti classe 1 (FUORI)
    y_pred = np.where(dist_sq_test <= z_final, 0, 1)
    
    # Calcolo metriche Macro (media non pesata per gestire classi sbilanciate)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print("\n" + "="*60)
    print(" ADDESTRAMENTO COMPLETATO!")
    print(f" Tempo di esecuzione: {tempo_totale:.2f} secondi")
    print(f" Iterazioni effettuate: {iterazione}")
    print(f" Budget totale speso: {budget_consumato:.2f}")
    print(f" Richieste effettuate all'oracolo: {richieste}")
    
    if len(storico_accuratezze) > 0:
        print("\n --- PRESTAZIONI FINALI (TEST SET) ---")
        print(f" Accuratezza (Accuracy):  {storico_accuratezze[-1]*100:.2f}%")
        print(f" Precision (Macro):       {precision*100:.2f}%")
        print(f" Recall (Macro):          {recall*100:.2f}%")
        print(f" F1-Score (Macro):        {f1*100:.2f}%")
        
        print("\n --- EVOLUZIONE ACCURATEZZA (LEARNING CURVE) ---")
        for i, acc in enumerate(storico_accuratezze):
            print(f"   Iter {i+1}: {acc*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()