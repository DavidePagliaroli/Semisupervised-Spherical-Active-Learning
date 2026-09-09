import warnings
# Ignora avvisi intrusivi del solutore
warnings.filterwarnings("ignore", message=".*SCS returned 2.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*qpsolvers.*")

import numpy as np
import time as tm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score 

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
    #X_full, y_full = prepara_breast_cancer()
    #X_full, y_full = prepara_heart()
    #X_full, y_full = prepara_ionosphere()
    X_full, y_full = prepara_spambase()
    #X_full, y_full = prepara_pima()
    
    valori_unici, conteggi = np.unique(y_full, return_counts=True)
    classe_minoritaria_orig = valori_unici[np.argmin(conteggi)]
    classe_maggioritaria_orig = valori_unici[np.argmax(conteggi)]
    
    print(f" [Topologia] Classe Minoritaria '{classe_minoritaria_orig}' -> mappata a 0 (Set A, DENTRO)")
    print(f" [Topologia] Classe Maggioritaria '{classe_maggioritaria_orig}' -> mappata a 1 (Set B, FUORI)")
    
    y_full = np.where(y_full == classe_minoritaria_orig, 0, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.30, random_state=42, stratify=y_full
    )

    print(f" [Dataset] Training Set: {X_train.shape[0]} campioni")
    print(f" [Dataset] Test Set: {X_test.shape[0]} campioni")
    """
    # ---------------------------------------------------------
    # 1. SELEZIONE E PREPARAZIONE DATI
    # ---------------------------------------------------------
    
    # 1. Recupero dati grezzi e pipeline non ancora addestrata
    #X_raw, y_full, pipeline = prepara_breast_cancer()
    X_raw, y_full, pipeline = prepara_heart()
    #X_raw, y_full, pipeline = prepara_ionosphere()
    #X_raw, y_full, pipeline = prepara_spambase()
    #X_raw, y_full, pipeline = prepara_pima()
    
    valori_unici, conteggi = np.unique(y_full, return_counts=True)
    classe_minoritaria_orig = valori_unici[np.argmin(conteggi)]
    classe_maggioritaria_orig = valori_unici[np.argmax(conteggi)]
    
    print(f" [Topologia] Classe Minoritaria '{classe_minoritaria_orig}' -> mappata a 0 (Set A, DENTRO)")
    print(f" [Topologia] Classe Maggioritaria '{classe_maggioritaria_orig}' -> mappata a 1 (Set B, FUORI)")
    
    y_full = np.where(y_full == classe_minoritaria_orig, 0, 1)
    
    # 2. Split sui dati ancora GREZZI (Nessuna statistica globale calcolata)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_full, test_size=0.3, stratify=y_full, random_state=42
    )
    
    # 3. APPLICAZIONE DELLA PIPELINE (Rimozione Data Leakage)
    print(" [Pre-processing] Addestramento scaler/imputer SOLO sul Training Set...")
    # Il fit_transform calcola media/varianza SOLO sul 70% dei dati e li scala
    X_train = pipeline.fit_transform(X_train_raw)
    
    # Il transform usa le regole imparate sopra per scalare il 30% dei dati di test (senza sbirciare)
    X_test = pipeline.transform(X_test_raw)

    print(f" [Dataset] Training Set: {X_train.shape[0]} campioni (Standardizzati)")
    print(f" [Dataset] Test Set: {X_test.shape[0]} campioni (Standardizzati)")
    """
    # ---------------------------------------------------------
    # 2. INIZIALIZZAZIONE ACTIVE LEARNING (COLD START)
    # ---------------------------------------------------------
    PERCENTUALE_INIZIALE = 0.2  #0.2
    
    X_seed, X_unlabeled, y_seed, Oracolo_Truth = train_test_split(
        X_train, y_train, 
        train_size=PERCENTUALE_INIZIALE, 
        random_state=42, 
        stratify=y_train
    )
    
    print(f" [Cold Start] Svelato il {PERCENTUALE_INIZIALE*100:.0f}% del Training Set ({len(X_seed)} campioni stratificati).")

    A = X_seed[y_seed == 0]
    B = X_seed[y_seed == 1]
    
    if len(A) == 0 or len(B) == 0:
        raise ValueError("Il campionamento iniziale non contiene almeno un rappresentante per ciascuna classe.")

    # Parametri Active Learning
    richieste = 0
    MAX_ROUND = 15
    BUDGET_GLOBALE = 700
    BUDGET_PER_ITERAZIONE = 80.0
    c_base = 1.0
    W = 4.0
    MAX_SAMPLE = 60
    C1 = 0.01
    C2 = 0.01
    budget_consumato = 0.0
    iterazione = 1
    storico_accuratezze = [] # (accuratezza, numero_campioni_etichettati)
    tempo_inizio = tm.time()

    # ---------------------------------------------------------
    # 3. MODELLO INIZIALE (PRIMA DI QUALSIASI ITERAZIONE)
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print(" FASE INIZIALE: CALCOLO MODELLO BASE (ZERO ITERAZIONI)")
    print("-"*60)
    totale_etichettati = len(A) + len(B)
    
    prob = ProblemaSferico(A, B, X_unlabeled, C1=C1, C2=C2) 
    DADC(prob)
    sfera_ottimizzata = prob.Xk 
    memoria_sfera = np.copy(sfera_ottimizzata)
    
    acc_iniziale = calcola_accuratezza(sfera_ottimizzata, X_test, y_test, 0, 1)
    storico_accuratezze.append((acc_iniziale, totale_etichettati))
    
    print(f" -> ACCURATEZZA INIZIALE (addestrato su {totale_etichettati} elementi seed): {acc_iniziale*100:.2f}%")
    print("-"*60)

    # ---------------------------------------------------------
    # 4. CICLO DI ACTIVE LEARNING
    # ---------------------------------------------------------

    cali_consecutivi = 0

    while budget_consumato < BUDGET_GLOBALE and iterazione <= MAX_ROUND:
        budget_rimanente = BUDGET_GLOBALE - budget_consumato

        if len(X_unlabeled) == 0 or budget_rimanente < c_base:
            break
            
        print(f"\n[Iter {iterazione}] Budget Consumato: {budget_consumato:.2f}/{BUDGET_GLOBALE:.2f}")

        # --- A. SELEZIONE CON ZAINO (basata sulla sfera attuale) ---
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

        # --- B. EARLY STOPPING (Margine o Budget) ---
        if len(indici_scelti) == 0:
            print("  [STOP ANTICIPATO] Nessun punto informativo selezionabile. Modello sicuro o budget insufficiente.")
            break

        # --- C. AGGIORNAMENTO ORACOLO ---
        num_selezionati = len(indici_scelti)
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
            
        totale_etichettati = len(A) + len(B)
        budget_consumato += costo_speso
        print(f"  [Oracolo] Selezionati {num_selezionati} nuovi elementi. (Totale etichettati salito a: {totale_etichettati})")

        # --- D. NUOVO ADDESTRAMENTO SUI DATI AGGIORNATI ---
        prob = ProblemaSferico(A, B, X_unlabeled, C1=C1, C2=C2, start_v=memoria_sfera) 
        DADC(prob)
        sfera_ottimizzata = prob.Xk 
        memoria_sfera = np.copy(sfera_ottimizzata)

        # --- E. VALUTAZIONE DOPO L'AGGIUNTA DEI DATI ---
        acc = calcola_accuratezza(sfera_ottimizzata, X_test, y_test, 0, 1)
        storico_accuratezze.append((acc, totale_etichettati))
        print(f"  [Metriche] ACCURATEZZA ITERAZIONE {iterazione}: {acc*100:.2f}%")
        
        # --- F. CONTROLLO DEGRADO PRESTAZIONI (Patience = 2) ---
        # Controlliamo se l'accuratezza è scesa rispetto al round precedente
        if len(storico_accuratezze) >= 2:
            acc_corrente = storico_accuratezze[-1][0]
            acc_precedente = storico_accuratezze[-2][0]
            
            if acc_corrente <= acc_precedente:
                cali_consecutivi += 1
            else:
                cali_consecutivi = 0 # Reset se migliora o rimane stabile
                
            if cali_consecutivi >= 2:
                print(f"  [STOP ANTICIPATO] Accuratezza in calo per {cali_consecutivi} iterazioni consecutive. Arresto per prevenire l'Outlier Chasing.")
                break
        
        iterazione += 1

    # ---------------------------------------------------------
    # 5. RISULTATI FINALI E METRICHE GLOBALI
    # ---------------------------------------------------------
    tempo_totale = tm.time() - tempo_inizio
    
    # Ricostruiamo la geometria finale
    x0_final = np.asarray(sfera_ottimizzata[:-2], dtype=float)
    z_final = float(sfera_ottimizzata[-2])
    dist_sq_test = np.sum((X_test - x0_final)**2, axis=1)
    y_pred = np.where(dist_sq_test <= z_final, 0, 1)
    
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print("\n" + "="*60)
    print(" ADDESTRAMENTO COMPLETATO!")
    print(f" Tempo di esecuzione: {tempo_totale:.2f} secondi")
    print(f" Iterazioni effettuate: {iterazione - 1}")
    print(f" Budget totale speso: {budget_consumato:.2f}")
    print(f" Richieste cumulative all'oracolo: {richieste}")
    print(f" Punti non etichettati rimasti (|X|): {len(X_unlabeled)}")
    
    if len(storico_accuratezze) > 0:
        acc_finale, campioni_finali = storico_accuratezze[-1]
        print("\n --- PRESTAZIONI FINALI (TEST SET) ---")
        print(f" Accuratezza (Accuracy):  {acc_finale*100:.2f}%")
        print(f" Precision (Macro):       {precision*100:.2f}%")
        print(f" Recall (Macro):          {recall*100:.2f}%")
        print(f" F1-Score (Macro):        {f1*100:.2f}%")
        
        print("\n --- EVOLUZIONE ACCURATEZZA (LEARNING CURVE) ---")
        for i, (acc, n_campioni) in enumerate(storico_accuratezze):
            if i == 0:
                print(f"   Iniziale : {acc*100:.2f}%  (Addestrato su {n_campioni} elementi del seed)")
            else:
                print(f"   Iter {i:<4}: {acc*100:.2f}%  (Addestrato su {n_campioni} elementi totali)")
    print("="*60)

if __name__ == "__main__":
    main()