# numpy_models.py
# Reescribo los modelos del M4 en numpy para que el dashboard no dependa de TF
# (en la laptop de la facu no tengo TF instalado y cada vez que actualizo
# algo se me rompe). Basicamente replico:
#   - el MLP (dense 128 -> 64 -> salida)
#   - una RNN chica que uso como stand-in de la LSTM; la LSTM real esta
#     en el notebook, entrenada con Keras.
#
# El objetivo es producir los .npz y los .json que el dashboard lee, y
# algunas figuras de entrenamiento/confusion.
#
# ojo: este script tarda como 2-3 minutos en mi maquina.

import os, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
MODELS = os.path.join(HERE, "models")
os.makedirs(MODELS, exist_ok=True)

# -----------------------------------------------------------------
# Carga y preparación
# -----------------------------------------------------------------
df = pd.read_csv(os.path.join(DATA, "infocana_processed.csv"))
with open(os.path.join(DATA, "meta.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)

feature_cols = meta["feature_cols"]
TARGET_REG = "eficiencia_en_fabrica"
TARGET_CLS = "class_id"

# Split 70/15/15 estratificado por ingenio (evita fuga entre semanas del mismo ingenio)
ingenios = df["ingenio"].unique()
rng = np.random.default_rng(42)
# Estratificar ingenios por su class_id (que es consistente por ingenio)
ing_class = df.groupby("ingenio")["class_id"].first()
train_ings, val_ings, test_ings = [], [], []
for c in sorted(ing_class.unique()):
    ings_c = ing_class[ing_class == c].index.to_numpy()
    rng.shuffle(ings_c)
    n = len(ings_c)
    n_train = max(1, int(round(0.7 * n)))
    n_val = max(1, int(round(0.15 * n))) if n >= 3 else 0
    train_ings += list(ings_c[:n_train])
    val_ings += list(ings_c[n_train:n_train + n_val])
    test_ings += list(ings_c[n_train + n_val:])

train_mask = df["ingenio"].isin(train_ings)
val_mask = df["ingenio"].isin(val_ings)
test_mask = df["ingenio"].isin(test_ings)

print(f"Split: train={train_mask.sum()}  val={val_mask.sum()}  test={test_mask.sum()}")

# Normalización z-score (fit en train)
X_full = df[feature_cols].values.astype(np.float32)
mu = X_full[train_mask].mean(axis=0); sd = X_full[train_mask].std(axis=0) + 1e-9
Xn = (X_full - mu) / sd

y_reg = df[TARGET_REG].values.astype(np.float32)
y_reg_mu = y_reg[train_mask].mean(); y_reg_sd = y_reg[train_mask].std() + 1e-9
y_reg_n = (y_reg - y_reg_mu) / y_reg_sd

y_cls = df[TARGET_CLS].values.astype(np.int64)
n_classes = int(y_cls.max()) + 1

# Guardar scaler para dashboard
np.savez(os.path.join(MODELS, "scaler.npz"),
         mu=mu, sd=sd, y_reg_mu=y_reg_mu, y_reg_sd=y_reg_sd,
         feature_cols=np.array(feature_cols))

X_tr, y_tr_c, y_tr_r = Xn[train_mask], y_cls[train_mask], y_reg_n[train_mask]
X_va, y_va_c, y_va_r = Xn[val_mask],   y_cls[val_mask],   y_reg_n[val_mask]
X_te, y_te_c, y_te_r = Xn[test_mask],  y_cls[test_mask],  y_reg_n[test_mask]

# -----------------------------------------------------------------
# MLP (2 hidden layers: 128 -> 64) con ReLU + dropout + softmax/linear
# -----------------------------------------------------------------
def relu(x): return np.maximum(0, x)
def drelu(x): return (x > 0).astype(x.dtype)
def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def init_mlp(in_dim, hidden=(128, 64), out_dim=3, seed=0):
    rng = np.random.default_rng(seed)
    dims = [in_dim] + list(hidden) + [out_dim]
    W, b = [], []
    for i in range(len(dims) - 1):
        # He init
        W.append(rng.standard_normal((dims[i], dims[i+1])).astype(np.float32) * np.sqrt(2.0/dims[i]))
        b.append(np.zeros((dims[i+1],), dtype=np.float32))
    return W, b

def forward_mlp(W, b, X, training=False, drop=0.2, rng=None):
    a = X
    acts = [a]; pre = []; masks = []
    for i in range(len(W) - 1):
        z = a @ W[i] + b[i]; pre.append(z)
        a = relu(z)
        if training and drop > 0 and rng is not None:
            mask = (rng.random(a.shape) > drop).astype(a.dtype) / (1.0 - drop)
            a = a * mask
            masks.append(mask)
        else:
            masks.append(None)
        acts.append(a)
    z = a @ W[-1] + b[-1]; pre.append(z); acts.append(z); masks.append(None)
    return acts, pre, masks

def train_mlp(X_tr, y_tr, X_va, y_va, task="clf",
              hidden=(128, 64), lr=1e-3, epochs=40, batch=64, drop=0.2, seed=0):
    rng = np.random.default_rng(seed)
    in_dim = X_tr.shape[1]
    if task == "clf":
        out_dim = int(y_tr.max()) + 1
    else:
        out_dim = 1
        y_tr = y_tr.reshape(-1, 1)
        y_va = y_va.reshape(-1, 1)
    W, b = init_mlp(in_dim, hidden=hidden, out_dim=out_dim, seed=seed)
    # Adam state
    mW = [np.zeros_like(w) for w in W]; vW = [np.zeros_like(w) for w in W]
    mb = [np.zeros_like(bi) for bi in b]; vb = [np.zeros_like(bi) for bi in b]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    hist = {"loss": [], "val_loss": [], "metric": [], "val_metric": []}
    n = X_tr.shape[0]

    for ep in range(epochs):
        idx = rng.permutation(n)
        losses = []
        for i in range(0, n, batch):
            ii = idx[i:i+batch]
            xb = X_tr[ii]; yb = y_tr[ii]
            acts, pre, masks = forward_mlp(W, b, xb, training=True, drop=drop, rng=rng)
            if task == "clf":
                probs = softmax(acts[-1])
                # cross-entropy
                m = xb.shape[0]
                nll = -np.log(probs[np.arange(m), yb] + 1e-9).mean()
                losses.append(nll)
                dL = probs.copy(); dL[np.arange(m), yb] -= 1.0; dL /= m
            else:
                pred = acts[-1]
                diff = (pred - yb)
                mse = (diff**2).mean()
                losses.append(mse)
                dL = 2 * diff / (xb.shape[0] * pred.shape[1])
            # backward
            grads_W, grads_b = [None]*len(W), [None]*len(b)
            dA = dL
            for li in range(len(W)-1, -1, -1):
                a_prev = acts[li]
                grads_W[li] = a_prev.T @ dA
                grads_b[li] = dA.sum(axis=0)
                if li > 0:
                    dA_prev = dA @ W[li].T
                    # dropout mask applied to a_prev output -> passes through
                    if masks[li-1] is not None:
                        dA_prev = dA_prev * masks[li-1]
                    # relu derivative
                    dA_prev = dA_prev * drelu(pre[li-1])
                    dA = dA_prev
            # Adam step
            step += 1
            for li in range(len(W)):
                mW[li] = beta1*mW[li] + (1-beta1)*grads_W[li]
                vW[li] = beta2*vW[li] + (1-beta2)*(grads_W[li]**2)
                mhat = mW[li] / (1 - beta1**step); vhat = vW[li] / (1 - beta2**step)
                W[li] -= lr * mhat / (np.sqrt(vhat) + eps)
                mb[li] = beta1*mb[li] + (1-beta1)*grads_b[li]
                vb[li] = beta2*vb[li] + (1-beta2)*(grads_b[li]**2)
                mhat = mb[li] / (1 - beta1**step); vhat = vb[li] / (1 - beta2**step)
                b[li] -= lr * mhat / (np.sqrt(vhat) + eps)

        # Eval
        tr_metric, tr_loss = eval_mlp(W, b, X_tr, y_tr.squeeze() if task=="clf" else y_tr, task)
        va_metric, va_loss = eval_mlp(W, b, X_va, y_va.squeeze() if task=="clf" else y_va, task)
        hist["loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        hist["metric"].append(tr_metric); hist["val_metric"].append(va_metric)
        if ep % 5 == 0 or ep == epochs-1:
            label = "acc" if task=="clf" else "R²"
            print(f"  MLP/{task}  ep={ep+1:02d}  loss={tr_loss:.4f}  val_loss={va_loss:.4f}  {label}={tr_metric:.3f}  val_{label}={va_metric:.3f}")
    return (W, b), hist

def eval_mlp(W, b, X, y, task):
    acts, _, _ = forward_mlp(W, b, X, training=False)
    if task == "clf":
        probs = softmax(acts[-1])
        pred = probs.argmax(axis=1)
        acc = (pred == y).mean()
        nll = -np.log(probs[np.arange(len(y)), y] + 1e-9).mean()
        return float(acc), float(nll)
    else:
        pred = acts[-1].reshape(-1)
        y = y.reshape(-1)
        mse = float(((pred - y)**2).mean())
        ss_res = ((y - pred)**2).sum()
        ss_tot = ((y - y.mean())**2).sum() + 1e-9
        r2 = float(1 - ss_res / ss_tot)
        return r2, mse

# -----------------------------------------------------------------
# RNN (SimpleRNN, serves as LSTM stand-in) sobre secuencias por ingenio
# -----------------------------------------------------------------
def build_sequences(df, mask, feature_cols, target_reg_col, target_cls_col,
                    seq_len=8, mu=None, sd=None, y_mu=None, y_sd=None):
    """Por cada ingenio en mask, construye ventanas deslizantes de seq_len
    semanas consecutivas; target = valores de la ULTIMA semana."""
    Xs, yc, yr = [], [], []
    sub = df[mask].sort_values(["ingenio", "semana"])
    for _, g in sub.groupby("ingenio"):
        Xv = (g[feature_cols].values.astype(np.float32) - mu) / sd
        yv_c = g[target_cls_col].values.astype(np.int64)
        yv_r = (g[target_reg_col].values.astype(np.float32) - y_mu) / y_sd
        for i in range(len(g) - seq_len + 1):
            Xs.append(Xv[i:i+seq_len])
            yc.append(yv_c[i+seq_len-1])
            yr.append(yv_r[i+seq_len-1])
    return (np.asarray(Xs, dtype=np.float32),
            np.asarray(yc, dtype=np.int64),
            np.asarray(yr, dtype=np.float32))

SEQ_LEN = 8  # 8 semanas como ventana temporal (≈2 meses)

def tanh(x): return np.tanh(x)
def dtanh_y(y): return 1 - y*y

def init_rnn(in_dim, hidden=32, out_dim=3, seed=0):
    rng = np.random.default_rng(seed)
    Wx = rng.standard_normal((in_dim, hidden)).astype(np.float32) * np.sqrt(1.0/in_dim)
    Wh = rng.standard_normal((hidden, hidden)).astype(np.float32) * np.sqrt(1.0/hidden)
    bh = np.zeros((hidden,), dtype=np.float32)
    Wy = rng.standard_normal((hidden, out_dim)).astype(np.float32) * np.sqrt(1.0/hidden)
    by = np.zeros((out_dim,), dtype=np.float32)
    return {"Wx":Wx, "Wh":Wh, "bh":bh, "Wy":Wy, "by":by}

def rnn_forward(p, X):
    # X: (B, T, F)
    B, T, F = X.shape
    H = p["Wh"].shape[0]
    hs = np.zeros((B, T+1, H), dtype=np.float32)
    for t in range(T):
        hs[:, t+1] = tanh(X[:, t] @ p["Wx"] + hs[:, t] @ p["Wh"] + p["bh"])
    out = hs[:, -1] @ p["Wy"] + p["by"]
    return out, hs

def rnn_backward(p, X, hs, dOut):
    # dOut: (B, out_dim)
    B, T, F = X.shape
    H = p["Wh"].shape[0]
    gWx = np.zeros_like(p["Wx"]); gWh = np.zeros_like(p["Wh"]); gbh = np.zeros_like(p["bh"])
    gWy = hs[:, -1].T @ dOut
    gby = dOut.sum(axis=0)
    dh_next = dOut @ p["Wy"].T  # (B, H)
    for t in range(T-1, -1, -1):
        h = hs[:, t+1]
        dtanh = dh_next * dtanh_y(h)
        gWx += X[:, t].T @ dtanh
        gWh += hs[:, t].T @ dtanh
        gbh += dtanh.sum(axis=0)
        dh_next = dtanh @ p["Wh"].T
    return {"Wx":gWx, "Wh":gWh, "bh":gbh, "Wy":gWy, "by":gby}

def train_rnn(X_tr, y_tr, X_va, y_va, task="clf", hidden=32, lr=5e-3,
              epochs=30, batch=32, seed=0):
    rng = np.random.default_rng(seed)
    in_dim = X_tr.shape[-1]
    out_dim = int(y_tr.max()) + 1 if task == "clf" else 1
    p = init_rnn(in_dim, hidden=hidden, out_dim=out_dim, seed=seed)
    # Adam state
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v2) for k, v2 in p.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8; step = 0

    hist = {"loss": [], "val_loss": [], "metric": [], "val_metric": []}
    n = X_tr.shape[0]
    for ep in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch):
            ii = idx[i:i+batch]
            xb = X_tr[ii]; yb = y_tr[ii]
            out, hs = rnn_forward(p, xb)
            if task == "clf":
                probs = softmax(out)
                mb = xb.shape[0]
                dOut = probs.copy(); dOut[np.arange(mb), yb] -= 1.0; dOut /= mb
            else:
                yb = yb.reshape(-1, 1)
                dOut = 2 * (out - yb) / (xb.shape[0])
            grads = rnn_backward(p, xb, hs, dOut)
            step += 1
            for k in p:
                m[k] = beta1*m[k] + (1-beta1)*grads[k]
                v[k] = beta2*v[k] + (1-beta2)*(grads[k]**2)
                mhat = m[k]/(1-beta1**step); vhat = v[k]/(1-beta2**step)
                p[k] -= lr * mhat / (np.sqrt(vhat) + eps)
        tr_metric, tr_loss = eval_rnn(p, X_tr, y_tr, task)
        va_metric, va_loss = eval_rnn(p, X_va, y_va, task)
        hist["loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        hist["metric"].append(tr_metric); hist["val_metric"].append(va_metric)
        if ep % 5 == 0 or ep == epochs-1:
            label = "acc" if task=="clf" else "R²"
            print(f"  RNN/{task}  ep={ep+1:02d}  loss={tr_loss:.4f}  val_loss={va_loss:.4f}  {label}={tr_metric:.3f}  val_{label}={va_metric:.3f}")
    return p, hist

def eval_rnn(p, X, y, task):
    out, _ = rnn_forward(p, X)
    if task == "clf":
        probs = softmax(out)
        pred = probs.argmax(axis=1)
        acc = (pred == y).mean()
        nll = -np.log(probs[np.arange(len(y)), y] + 1e-9).mean()
        return float(acc), float(nll)
    else:
        y = y.reshape(-1)
        pred = out.reshape(-1)
        mse = float(((pred - y)**2).mean())
        ss_res = ((y - pred)**2).sum()
        ss_tot = ((y - y.mean())**2).sum() + 1e-9
        r2 = float(1 - ss_res / ss_tot)
        return r2, mse

# ================================================================
# ENTRENAMIENTO
# ================================================================
print("\n== Entrenando MLP clasificación ==")
(W_clf, b_clf), hist_mlp_clf = train_mlp(X_tr, y_tr_c, X_va, y_va_c,
                                         task="clf", hidden=(128, 64),
                                         lr=1e-3, epochs=40, batch=64, drop=0.2)

print("\n== Entrenando MLP regresión ==")
(W_reg, b_reg), hist_mlp_reg = train_mlp(X_tr, y_tr_r, X_va, y_va_r,
                                         task="reg", hidden=(128, 64),
                                         lr=1e-3, epochs=40, batch=64, drop=0.2)

# RNN sequences
print("\nConstruyendo secuencias para RNN (ventana=8 semanas)…")
Xs_tr, yc_tr, yr_tr = build_sequences(df, train_mask, feature_cols,
                                       TARGET_REG, TARGET_CLS,
                                       seq_len=SEQ_LEN, mu=mu, sd=sd,
                                       y_mu=y_reg_mu, y_sd=y_reg_sd)
Xs_va, yc_va, yr_va = build_sequences(df, val_mask, feature_cols,
                                       TARGET_REG, TARGET_CLS,
                                       seq_len=SEQ_LEN, mu=mu, sd=sd,
                                       y_mu=y_reg_mu, y_sd=y_reg_sd)
Xs_te, yc_te, yr_te = build_sequences(df, test_mask, feature_cols,
                                       TARGET_REG, TARGET_CLS,
                                       seq_len=SEQ_LEN, mu=mu, sd=sd,
                                       y_mu=y_reg_mu, y_sd=y_reg_sd)
print(f"RNN shapes: tr={Xs_tr.shape}  va={Xs_va.shape}  te={Xs_te.shape}")

print("\n== Entrenando RNN clasificación ==")
p_clf, hist_rnn_clf = train_rnn(Xs_tr, yc_tr, Xs_va, yc_va,
                                task="clf", hidden=32, lr=5e-3,
                                epochs=30, batch=32)
print("\n== Entrenando RNN regresión ==")
p_reg, hist_rnn_reg = train_rnn(Xs_tr, yr_tr, Xs_va, yr_va,
                                task="reg", hidden=32, lr=5e-3,
                                epochs=30, batch=32)

# ================================================================
# Evaluación en test
# ================================================================
def cls_metrics(y_true, y_pred, n_cls=3):
    acc = float((y_true == y_pred).mean())
    f1s = []
    for c in range(n_cls):
        tp = int(((y_pred==c) & (y_true==c)).sum())
        fp = int(((y_pred==c) & (y_true!=c)).sum())
        fn = int(((y_pred!=c) & (y_true==c)).sum())
        prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        f1s.append(f1)
    return acc, float(np.mean(f1s)), f1s

def reg_metrics(y_true_n, y_pred_n, y_mu, y_sd):
    # denormalize
    y_true = y_true_n * y_sd + y_mu
    y_pred = y_pred_n * y_sd + y_mu
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2) + 1e-9
    r2 = float(1 - ss_res/ss_tot)
    return mae, rmse, r2

# MLP clf test
acts, _, _ = forward_mlp(W_clf, b_clf, X_te, training=False)
mlp_clf_pred = softmax(acts[-1]).argmax(axis=1)
mlp_acc, mlp_f1, mlp_f1_per = cls_metrics(y_te_c, mlp_clf_pred)
# MLP reg test
acts_r, _, _ = forward_mlp(W_reg, b_reg, X_te, training=False)
mlp_reg_pred = acts_r[-1].reshape(-1)
mlp_mae, mlp_rmse, mlp_r2 = reg_metrics(y_te_r, mlp_reg_pred, y_reg_mu, y_reg_sd)

# RNN clf test
out_c, _ = rnn_forward(p_clf, Xs_te)
rnn_clf_pred = softmax(out_c).argmax(axis=1)
rnn_acc, rnn_f1, rnn_f1_per = cls_metrics(yc_te, rnn_clf_pred)
# RNN reg test
out_r, _ = rnn_forward(p_reg, Xs_te)
rnn_reg_pred = out_r.reshape(-1)
rnn_mae, rnn_rmse, rnn_r2 = reg_metrics(yr_te, rnn_reg_pred, y_reg_mu, y_reg_sd)

metrics = {
    "MLP": {
        "classification": {"accuracy": mlp_acc, "f1_macro": mlp_f1, "f1_per_class": mlp_f1_per},
        "regression":     {"MAE": mlp_mae, "RMSE": mlp_rmse, "R2": mlp_r2},
    },
    "LSTM_like": {
        "classification": {"accuracy": rnn_acc, "f1_macro": rnn_f1, "f1_per_class": rnn_f1_per},
        "regression":     {"MAE": rnn_mae, "RMSE": rnn_rmse, "R2": rnn_r2},
    },
    "test_sizes": {"tabular": int(X_te.shape[0]), "seq": int(Xs_te.shape[0])},
}
with open(os.path.join(MODELS, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

history = {
    "MLP_clf": hist_mlp_clf, "MLP_reg": hist_mlp_reg,
    "RNN_clf": hist_rnn_clf, "RNN_reg": hist_rnn_reg,
}
with open(os.path.join(MODELS, "history.json"), "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2, default=float)

# Guardar pesos
np.savez(os.path.join(MODELS, "mlp_clf.npz"), *W_clf, *b_clf)
np.savez(os.path.join(MODELS, "mlp_reg.npz"), *W_reg, *b_reg)
np.savez(os.path.join(MODELS, "rnn_clf.npz"), **p_clf)
np.savez(os.path.join(MODELS, "rnn_reg.npz"), **p_reg)

# ================================================================
# Gráficas
# ================================================================
plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

def plot_curves(hist, title, ylabel_metric, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist["loss"], label="train", color="#028090")
    axes[0].plot(hist["val_loss"], label="val", color="#B85042", linestyle="--")
    axes[0].set_title(f"{title} — Loss"); axes[0].set_xlabel("Época"); axes[0].legend()
    axes[1].plot(hist["metric"], label="train", color="#028090")
    axes[1].plot(hist["val_metric"], label="val", color="#B85042", linestyle="--")
    axes[1].set_title(f"{title} — {ylabel_metric}"); axes[1].set_xlabel("Época"); axes[1].legend()
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()

plot_curves(hist_mlp_clf, "MLP — Clasificación", "Accuracy",
            os.path.join(FIG, "10_curvas_mlp_clf.png"))
plot_curves(hist_mlp_reg, "MLP — Regresión", "R²",
            os.path.join(FIG, "11_curvas_mlp_reg.png"))
plot_curves(hist_rnn_clf, "LSTM-like (RNN) — Clasificación", "Accuracy",
            os.path.join(FIG, "12_curvas_rnn_clf.png"))
plot_curves(hist_rnn_reg, "LSTM-like (RNN) — Regresión", "R²",
            os.path.join(FIG, "13_curvas_rnn_reg.png"))

# Confusion matrices
def plot_conf(y_true, y_pred, title, path, class_names):
    K = len(class_names)
    cm = np.zeros((K, K), dtype=int)
    for t, p_ in zip(y_true, y_pred):
        cm[int(t), int(p_)] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(class_names, rotation=20); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(title, fontweight="bold")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()

class_names = meta["class_names"]
plot_conf(y_te_c, mlp_clf_pred, "Matriz de confusión — MLP",
          os.path.join(FIG, "20_confusion_mlp.png"), class_names)
plot_conf(yc_te, rnn_clf_pred, "Matriz de confusión — LSTM-like (RNN)",
          os.path.join(FIG, "21_confusion_rnn.png"), class_names)

# Comparativa global
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
# Clasificación
labels_c = ["Accuracy", "F1 macro"]
mlp_vals = [mlp_acc, mlp_f1]
rnn_vals = [rnn_acc, rnn_f1]
x = np.arange(len(labels_c)); w = 0.35
axes[0].bar(x - w/2, mlp_vals, w, label="MLP", color="#028090", edgecolor="#222")
axes[0].bar(x + w/2, rnn_vals, w, label="LSTM-like", color="#B85042", edgecolor="#222")
axes[0].set_xticks(x); axes[0].set_xticklabels(labels_c)
axes[0].set_title("Clasificación — Test"); axes[0].set_ylim(0, 1.05); axes[0].legend()
for i, v in enumerate(mlp_vals): axes[0].text(i-w/2, v+0.02, f"{v:.2f}", ha="center", fontsize=9)
for i, v in enumerate(rnn_vals): axes[0].text(i+w/2, v+0.02, f"{v:.2f}", ha="center", fontsize=9)
# Regresión (barras con escalas distintas)
labels_r = ["MAE", "RMSE", "R²"]
mlp_r = [mlp_mae, mlp_rmse, mlp_r2]
rnn_r = [rnn_mae, rnn_rmse, rnn_r2]
ax2 = axes[1]
x = np.arange(len(labels_r))
# Mostrar como etiquetas numericas
ax2.axis("off")
ax2.table(cellText=[[f"{v:.3f}" for v in mlp_r],
                    [f"{v:.3f}" for v in rnn_r]],
          rowLabels=["MLP", "LSTM-like"],
          colLabels=labels_r, loc="center", cellLoc="center")
ax2.set_title("Regresión — Test", fontweight="bold")
plt.suptitle("Comparativa MLP vs LSTM-like en el conjunto test", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "30_comparativa_metricas.png"), bbox_inches="tight")
plt.close()

print("\n== Resumen final (test) ==")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print("\nFiguras guardadas en:", FIG)
print("Modelos guardados en:", MODELS)
                                                                                                                                                                                                                                                                                                                                                                                             