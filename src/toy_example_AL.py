import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# ------------------------------------------------------------------
# 1. GENERAZIONE DEL DATASET
# ------------------------------------------------------------------
N_PER_CLASS = 200          # 200 + 200 = 400 elementi totali
SIGMA = 2
MEAN_NEG = np.array([-2.0, 0.0])   # classe -1
MEAN_POS = np.array([2.0, 0.0])    # classe +1

X_neg = rng.normal(loc=MEAN_NEG, scale=SIGMA, size=(N_PER_CLASS, 2))
X_pos = rng.normal(loc=MEAN_POS, scale=SIGMA, size=(N_PER_CLASS, 2))

X = np.vstack([X_neg, X_pos])
y = np.concatenate([-np.ones(N_PER_CLASS), np.ones(N_PER_CLASS)])

X_pool, X_test, y_pool, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

# ------------------------------------------------------------------
# 2. VISUALIZZAZIONE DEL DATASET COMPLETO
# ------------------------------------------------------------------
def plot_classes(ax, X, y, colors=('tab:blue', 'tab:red'), alpha=0.7,
                  labels=('Classe -1', 'Classe +1')):
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c=colors[0], alpha=alpha,
               edgecolor='k', linewidth=0.3, label=labels[0])
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c=colors[1], alpha=alpha,
               edgecolor='k', linewidth=0.3, label=labels[1])

fig, ax = plt.subplots(figsize=(6, 6))
plot_classes(ax, X, y)
ax.set_xlabel("x1"); ax.set_ylabel("x2")
ax.legend(); ax.axis('equal'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("01_dataset.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# Funzione di supporto: disegna il confine decisionale di un modello
# ------------------------------------------------------------------
def plot_decision_boundary(model, X_train, y_train, X_bg=None, y_bg=None,
                            title="", filename=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    if X_bg is not None and len(X_bg) > 0:
        plot_classes(ax, X_bg, y_bg, colors=('lightsteelblue', 'lightsalmon'),
                     alpha=0.35, labels=('Pool -1 (non usati)', 'Pool +1 (non usati)'))

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                          np.linspace(y_min, y_max, 400))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()],
                colors=['#cfe0f7', '#f7d6c9'], alpha=0.5)
    ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)

    plot_classes(ax, X_train, y_train, colors=('tab:blue', 'tab:red'),
                 alpha=1.0, labels=('Training -1', 'Training +1'))

    ax.set_title(title)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(loc='upper left', fontsize=8)
    ax.axis('equal'); ax.grid(alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    plt.close()

# ------------------------------------------------------------------
# 3. PRIMO ADDESTRAMENTO: 20 ISTANZE CASUALI DAL POOL
# ------------------------------------------------------------------
n_initial = 20
idx_pool = np.arange(len(X_pool))
idx_initial = rng.choice(idx_pool, size=n_initial, replace=False)

X_train_1 = X_pool[idx_initial]
y_train_1 = y_pool[idx_initial]

model_1 = LogisticRegression()
model_1.fit(X_train_1, y_train_1)

y_pred_1 = model_1.predict(X_test)
acc_1 = accuracy_score(y_test, y_pred_1)
print(f"[Modello 1] Addestrato su {n_initial} punti casuali "
      f"- Accuratezza sul test set: {acc_1:.4f}")

idx_remaining = np.setdiff1d(idx_pool, idx_initial)
X_remaining = X_pool[idx_remaining]
y_remaining = y_pool[idx_remaining]

plot_decision_boundary(
    model_1, X_train_1, y_train_1, X_bg=X_remaining, y_bg=y_remaining,
    
    filename="02_modello_1.png"
)

# ------------------------------------------------------------------
# 4. ACTIVE LEARNING: LE 20 ISTANZE PIU' VICINE AL CONFINE
# ------------------------------------------------------------------

distances = np.abs(model_1.decision_function(X_remaining))
idx_closest_local = np.argsort(distances)[:20]  #MODIFICARE IL NUMERO DI ELEMENTI SELEZIONATI

X_new = X_remaining[idx_closest_local]
y_new = y_remaining[idx_closest_local]

X_train_2 = np.vstack([X_train_1, X_new])
y_train_2 = np.concatenate([y_train_1, y_new])

model_2 = LogisticRegression()
model_2.fit(X_train_2, y_train_2)

y_pred_2 = model_2.predict(X_test)
acc_2 = accuracy_score(y_test, y_pred_2)
print(f"[Modello 2] Riaddestrato su {len(X_train_2)} punti "
      f"(20 iniziali + 20 vicini al confine) - Accuratezza sul test set: {acc_2:.4f}")

idx_remaining_2 = np.setdiff1d(np.arange(len(X_remaining)), idx_closest_local)
X_remaining_2 = X_remaining[idx_remaining_2]
y_remaining_2 = y_remaining[idx_remaining_2]

plot_decision_boundary(
    model_2, X_train_2, y_train_2, X_bg=X_remaining_2, y_bg=y_remaining_2,
    filename="03_modello_2.png"
)

# ------------------------------------------------------------------
# RIEPILOGO
# ------------------------------------------------------------------
print("\n--- Riepilogo ---")
print(f"Accuratezza modello 1 (20 punti casuali):              {acc_1:.4f}")
print(f"Accuratezza modello 2 (40 punti, +20 vicini al confine): {acc_2:.4f}")