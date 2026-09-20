import warnings
# Ignora qualsiasi avviso che contenga "SCS returned 2" nel testo
warnings.filterwarnings("ignore", message=".*SCS returned 2.*")
# Ignora preventivamente tutti gli UserWarning che provengono dalla libreria qpsolvers
warnings.filterwarnings("ignore", category=UserWarning, module=".*qpsolvers.*")
import numpy as np
import time as tm
from sklearn.metrics import accuracy_score
from algormeter.libs import Kernel
from DADC import DADC


# ==========================================
# 1. LA CLASSE DEL PROBLEMA MATEMATICO 
# ==========================================
class ProblemaSferico(Kernel):
    def __init__(self, A, B, X, C1=1.0, C2=1.0, start_v=None):
        self.d = np.array(X).shape[1]
        self.dimension = self.d + 2
        super().__init__(self.dimension) 
        
        self.A = np.array(A) if len(A) > 0 else np.empty((0, self.d))
        self.B = np.array(B) if len(B) > 0 else np.empty((0, self.d))
        self.X = np.array(X)
        self.C1 = C1
        self.C2 = C2
        self.PENALTY = 1e5
       
        self.epsilon = 1e-2
        
        if start_v is not None:
            self.XStart = np.copy(start_v)
        else:
            centro_iniziale = np.mean(self.A, axis=0) if len(self.A) > 0 else np.zeros(self.d)
            z_iniziale = 2.5 
            q_iniziale = 1.5
            self.XStart = np.concatenate([centro_iniziale, [z_iniziale, q_iniziale]])

    def _unpack(self, v):
        x0 = np.asarray(v[:-2], dtype=float)
        z = float(v[-2])
        q = float(v[-1])
        return x0, z, q

    # ==========================================
    # METODI PRINCIPALI DEL DADC
    # ==========================================
    def _f1(self, v) -> float:
        x0, z, q = self._unpack(v)

        a = np.asarray(self.A, dtype=float)
        b = np.asarray(self.B, dtype=float)
        x = np.asarray(self.X, dtype=float)

        da = np.sum((a - x0) ** 2, axis=1)
        db = np.sum((b - x0) ** 2, axis=1)
        dx = np.sum((x - x0) ** 2, axis=1)

        term_a = self.C1 * np.sum(np.maximum(0.0, q - z + da))
        term_b = self.C1 * np.sum(np.maximum(q + z, db))
        term_x = self.C2 * np.sum(np.maximum(dx - z + q, np.maximum(0.0, 2 * (dx - z))))
        
        # --- PENALITA' ---
        vincolo_q_positivo = self.PENALTY * max(0.0, self.epsilon - q)  # Si attiva se q < epsilon
        vincolo_q_minore_z = self.PENALTY * max(0.0, q - z)             # Si attiva se q > z

        return -q + term_a + term_b + term_x + vincolo_q_positivo + vincolo_q_minore_z

    def _f2(self, v) -> float:
        x0, z, q = self._unpack(v)

        b = np.asarray(self.B, dtype=float)
        x = np.asarray(self.X, dtype=float)

        db = np.sum((b - x0) ** 2, axis=1)
        dx = np.sum((x - x0) ** 2, axis=1)

        term_b = self.C1 * np.sum(db)
        term_x = self.C2 * np.sum(np.maximum(0.0, 2 * (dx - z)))

        return term_b + term_x

    def _gf1(self, v):
        x0, z, q = self._unpack(v)

        a = np.asarray(self.A, dtype=float)
        b = np.asarray(self.B, dtype=float)
        x = np.asarray(self.X, dtype=float)

        grad_x0 = np.zeros_like(x0)
        grad_z = 0.0
        grad_q = -1.0 

        if len(a) > 0:
            da = np.sum((a - x0) ** 2, axis=1)
            active_a = (q - z + da) > 0
            grad_x0 += self.C1 * np.sum(2 * (x0 - a[active_a]), axis=0)
            grad_z += -self.C1 * np.sum(active_a)
            grad_q += self.C1 * np.sum(active_a)

        if len(b) > 0:
            db = np.sum((b - x0) ** 2, axis=1)
            branch_h = (q + z) >= db
            grad_x0 += self.C1 * np.sum(2 * (x0 - b[~branch_h]), axis=0)
            grad_z += self.C1 * np.sum(branch_h)
            grad_q += self.C1 * np.sum(branch_h)

        if len(x) > 0:
            dx = np.sum((x - x0) ** 2, axis=1)
            u = dx - z + q
            w = 2 * (dx - z)
            mask_u = u >= np.maximum(0.0, w)
            mask_w = (~mask_u) & (w > 0)

            grad_x0 += self.C2 * np.sum(2 * (x0 - x[mask_u]), axis=0)
            grad_z += -self.C2 * np.sum(mask_u)
            grad_q += self.C2 * np.sum(mask_u)

            grad_x0 += self.C2 * np.sum(4 * (x0 - x[mask_w]), axis=0)
            grad_z += -2 * self.C2 * np.sum(mask_w)

        if (self.epsilon - q) > 0:            
            grad_q -= self.PENALTY
        if (q - z) > 0:      
            grad_q += self.PENALTY
            grad_z -= self.PENALTY

        return np.concatenate([grad_x0, [grad_z, grad_q]])

    def _gf2(self, v):
       
        x0, z, q = self._unpack(v)

        b = np.asarray(self.B, dtype=float)
        x = np.asarray(self.X, dtype=float)

        grad_x0 = np.zeros_like(x0)
        grad_z = 0.0
        grad_q = 0.0  

        if len(b) > 0:
            grad_x0 += self.C1 * np.sum(2 * (x0 - b), axis=0)

        if len(x) > 0:
            dx = np.sum((x - x0) ** 2, axis=1)
            active = 2 * (dx - z) > 0
            
            grad_x0 += self.C2 * np.sum(4 * (x0 - x[active]), axis=0)
            grad_z += -2 * self.C2 * np.sum(active)

        return np.concatenate([grad_x0, [grad_z, grad_q]])

# ==========================================
# 2. LOGICA DELLO ZAINO (Knapsack)
# ==========================================

import numpy as np

def seleziona_con_zaino(sfera_v, X_unlabeled, budget_attuale, C2, c_base=1.0, W=4.0, max_elementi=30):
    x0 = np.asarray(sfera_v[:-2], dtype=float)
    z = float(sfera_v[-2])
    q = float(sfera_v[-1])
    R = (np.sqrt(max(0.0, z + q)) + np.sqrt(max(0.0, z - q))) / 2.0
    M = (np.sqrt(max(0.0, z + q)) - np.sqrt(max(0.0, z - q))) / 2.0
    print(f"Raggio effettivo R={R:.2f}, Margine effettivo M={M:.2f}")
   
    dist_sq = np.sum((X_unlabeled - x0)**2, axis=1)
    valori_vj = C2 * np.maximum(0.0, q - np.abs(dist_sq - z))
    
    v_max = np.max(valori_vj) if np.max(valori_vj) > 0 else 1.0
    costi_wj = c_base + W * (valori_vj / v_max)
    

    oggetti_validi = []
    
    for i, (v, w) in enumerate(zip(valori_vj, costi_wj)):
        if v > 1e-7: 
            oggetti_validi.append({'idx': i, 'v': v, 'w': w, 'd': v / w})
            
    print("\n--- [Zaino Cost-Aware] ---")
    print(f"Elementi nel margine (V_j > 0): {len(oggetti_validi)}")
    
    if len(oggetti_validi) == 0:
        return [], [], 0.0
        
    oggetti_validi.sort(key=lambda x: x['d'], reverse=True)
    
    
    if max_elementi is not None and len(oggetti_validi) > max_elementi:
        oggetti_validi = oggetti_validi[:max_elementi]
        
    n_items = len(oggetti_validi)
    print(f"Elementi inviati al Branch and Bound (Pre-screening top-{max_elementi}): {n_items}")
    

    miglior_valore = 0.0
    miglior_selezione = []
    
    def calcola_bound(livello, peso_corrente, valore_corrente):
        if peso_corrente >= budget_attuale:
            return valore_corrente
            
        bound = valore_corrente
        peso_tot = peso_corrente
        
        for j in range(livello, n_items):
            oggetto = oggetti_validi[j]
            if peso_tot + oggetto['w'] <= budget_attuale:
                peso_tot += oggetto['w']
                bound += oggetto['v']
            else:
                spazio_rimanente = budget_attuale - peso_tot
                bound += spazio_rimanente * oggetto['d']
                break
                
        return bound

    stack = [(0, 0.0, 0.0, [])]
    
    while stack:
        livello, peso_curr, val_curr, sel_curr = stack.pop()
        
        if livello == n_items:
            continue
            
        oggetto_corrente = oggetti_validi[livello]
        
        peso_con = peso_curr + oggetto_corrente['w']
        val_con = val_curr + oggetto_corrente['v']
        
        if peso_con <= budget_attuale:
            if val_con > miglior_valore:
                miglior_valore = val_con
                miglior_selezione = sel_curr + [oggetto_corrente['idx']]
            
            bound_con = calcola_bound(livello + 1, peso_con, val_con)
            if bound_con > miglior_valore:
                stack.append((livello + 1, peso_con, val_con, sel_curr + [oggetto_corrente['idx']]))
                
        bound_senza = calcola_bound(livello + 1, peso_curr, val_curr)
        
        if bound_senza > miglior_valore:
            stack.append((livello + 1, peso_curr, val_curr, sel_curr))

    indici_scelti = miglior_selezione
    valori_scelti = [valori_vj[i] for i in indici_scelti]
    costo_speso = sum([costi_wj[i] for i in indici_scelti])

    R = (np.sqrt(max(0.0, z + q)) + np.sqrt(max(0.0, z - q))) / 2.0
    M = (np.sqrt(max(0.0, z + q)) - np.sqrt(max(0.0, z - q))) / 2.0
    print(f"Raggio effettivo R={R:.2f}, Margine effettivo M={M:.2f}")
    
    return indici_scelti, valori_scelti, costo_speso

def seleziona_per_prossimità(sfera_v, X_unlabeled, budget_attuale, C2, c_base=1.0, W=4.0, max_elementi = 30):

    x0 = np.asarray(sfera_v[:-2], dtype=float)
    z = float(sfera_v[-2])
    q = float(sfera_v[-1])
    
    dist_sq = np.sum((X_unlabeled - x0)**2, axis=1)
    valori_vj = C2 * np.maximum(0.0, q - np.abs(dist_sq - z))
    
    v_max = np.max(valori_vj) if np.max(valori_vj) > 0 else 1.0
    costi_wj = c_base + W * (valori_vj / v_max)   
  
    oggetti_validi = []
    for i, (v, w) in enumerate(zip(valori_vj, costi_wj)):
        if v > 1e-7: 
            oggetti_validi.append({'idx': i, 'v': v, 'w': w})
            
    oggetti_ordinati = sorted(oggetti_validi, key=lambda x: x['v'], reverse=True)
    
    elementi_selezionati = []
    costo_totale = 0.0
    
    for obj in oggetti_ordinati:
        if costo_totale + obj['w'] <= budget_attuale:
            elementi_selezionati.append(obj['idx'])
            costo_totale += obj['w']
        else:
            break
            
    return elementi_selezionati, valori_vj, costo_totale

import numpy as np

import numpy as np

def seleziona_casualmente(sfera_v, X_unlabeled, budget_attuale, C2, c_base=1.0, W=4.0, max_elementi = 30):
    x0 = np.asarray(sfera_v[:-2], dtype=float)
    z = float(sfera_v[-2])
    q = float(sfera_v[-1])
    
    dist_sq = np.sum((X_unlabeled - x0)**2, axis=1)
    valori_vj = C2 * np.maximum(0.0, q - np.abs(dist_sq - z))
    
    v_max = np.max(valori_vj) if np.max(valori_vj) > 0 else 1.0
    costi_wj = c_base + W * (valori_vj / v_max)
    
    oggetti_validi = []
    for i, (v, w) in enumerate(zip(valori_vj, costi_wj)):
        if v > 1e-7: 
            oggetti_validi.append({'idx': i, 'v': v, 'w': w})
            
    np.random.shuffle(oggetti_validi)
    
    elementi_selezionati = []
    costo_totale = 0.0
    
    for obj in oggetti_validi:
        if costo_totale + obj['w'] <= budget_attuale:
            elementi_selezionati.append(obj['idx'])
            costo_totale += obj['w']
        else:
            break
            
    return elementi_selezionati, valori_vj, costo_totale

# ==========================================
# 3. CALCOLO ACCURATEZZA DEL MODELLO
# ==========================================
def calcola_accuratezza(sfera_v, X_test, y_test, classe_minoritaria, classe_maggioritaria):
    x0 = np.asarray(sfera_v[:-2], dtype=float)
    z = float(sfera_v[-2])
    q = float(sfera_v[-1])
    
    radice = np.sqrt(max(0.0, z**2 - q**2))
    R_sq = (z + radice) / 2.0
    
    dist_sq = np.sum((X_test - x0)**2, axis=1)
    
    y_pred = np.where(dist_sq <= R_sq, classe_minoritaria, classe_maggioritaria)
    
    return accuracy_score(y_test, y_pred)