"""
Composants partagés par les pipelines LSTM (CasinoLimit et Zeek) :
- architecture ``LSTMBinary`` (bidirectionnelle, masquage explicite du
  padding via ``pack_padded_sequence``) ;
- utilitaires d'entraînement / évaluation ;
- ``XNIDSExplainer``, l'explicabilité "Sparse Group Lasso" auto-contenue
  (ne dépend d'aucun fichier précalculé, ni du package pip externe
  ``xnids``), portée à l'identique sur les deux jeux de données — voir
  ``scripts/lstm/README.md``, section "Choix d'architecture", pour le
  détail de cette décision.

Ce module ne contient aucune information spécifique à un dataset : les
features, les groupes de features et les chemins sont définis dans
``casinolimit_data.py`` et ``zeek_data.py``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# MODÈLE
# ============================================================

class LSTMBinary(nn.Module):
    """LSTM bidirectionnel -> 1 logit. Le padding est ignoré par
    construction grâce à ``pack_padded_sequence`` (jamais vu par le
    LSTM, ni en forward ni en backward)."""

    def __init__(self, n_features, hidden_size=64, num_layers=2,
                 dropout=0.2, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        out_dim = hidden_size * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        return self.head(self.norm(h)).squeeze(-1)


class SequenceDataset(Dataset):
    """Dataset générique (X, longueur réelle, label)."""

    def __init__(self, X, lengths, y):
        self.X = torch.from_numpy(np.asarray(X)).float()
        self.lengths = torch.from_numpy(np.asarray(lengths)).long()
        self.y = torch.from_numpy(np.asarray(y)).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.lengths[idx], self.y[idx]


# ============================================================
# ENTRAÎNEMENT / ÉVALUATION
# ============================================================

def train_lstm(model, train_loader, device, n_epochs, pos_weight):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for xb, lb, yb in train_loader:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb, lb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{n_epochs} — loss = {avg_loss:.6f}")

    return model


def get_probs(model, loader, device):
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for xb, lb, yb in loader:
            xb, lb = xb.to(device), lb.to(device)
            logits = model(xb, lb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_true.append(yb.numpy())
    return np.concatenate(all_probs), np.concatenate(all_true).astype(int)


def find_best_threshold(model, val_loader, device):
    probs_val, y_val_true = get_probs(model, val_loader, device)
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_thr, best_score = 0.5, -1.0
    for thr in thresholds:
        preds = (probs_val >= thr).astype(int)
        score = balanced_accuracy_score(y_val_true, preds)
        if score > best_score:
            best_score, best_thr = score, thr
    print(f"🎯 Seuil optimal trouvé sur VALIDATION : {best_thr:.2f} "
          f"(Balanced Accuracy val = {best_score:.4f})")
    # float() natif : np.arange() donne des np.float64, et torch.load()
    # (PyTorch >= 2.6, weights_only=True par défaut) refuse de désérialiser
    # un numpy scalar non explicitement autorisé — bug trouvé en testant le
    # chargement réel des checkpoints (voir scripts/lstm/README.md).
    return float(best_thr)


def compute_metrics_with_loss(model, loader, device, criterion, threshold=0.5):
    """Version légère (pas d'impression du rapport complet), utilisée
    pour le suivi époque par époque pendant l'entraînement."""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for xb, lb, yb in loader:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            logits = model(xb, lb)
            loss = criterion(logits, yb)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }


def evaluate_lstm(model, loader, device, threshold=0.5):
    """Évaluation complète et verbeuse (rapport de classification +
    matrice de confusion) — utilisée pour le rapport final sur TEST."""
    probs, y_true = get_probs(model, loader, device)
    predictions = (probs >= threshold).astype(int)

    metrics = {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "ROC_AUC": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan"),
        "Threshold": float(threshold),
    }
    print(f"\nAccuracy          : {metrics['Accuracy']:.4f}")
    print(f"Precision         : {metrics['Precision']:.4f}")
    print(f"Recall            : {metrics['Recall']:.4f}")
    print(f"F1-score          : {metrics['F1']:.4f}")
    print(f"Balanced Accuracy : {metrics['Balanced_Accuracy']:.4f}")
    print(f"ROC-AUC           : {metrics['ROC_AUC']:.4f}")
    print("\nClassification Report :\n")
    print(classification_report(y_true, predictions, target_names=["BENIGN", "ATTACK"], zero_division=0))
    print("Matrice de confusion :")
    print(confusion_matrix(y_true, predictions))
    return metrics


# ============================================================
# XNIDS — explicabilité auto-contenue (Sparse Group Lasso)
# ============================================================
#
# Implémentation auto-contenue (aucune dépendance au package pip externe
# "xnids", qui expose une signature de constructeur différente et
# incompatible avec celle utilisée ici). C'est cette version —
# initialement développée et validée sur CasinoLimit — qui est réutilisée
# à l'identique pour Zeek, afin que les deux datasets disposent d'un
# pipeline xNIDS -> Agent 1 -> RAG -> Agent 2 entièrement fonctionnel et
# ne dépendant d'aucun fichier précalculé.

class XNIDSExplainer:
    def __init__(self, model, feature_names, group_dict, window, device,
                 num_samples=200, delta=0.02):
        self.model = model.eval()
        self.feature_names = feature_names
        self.group_dict = group_dict
        self.window = window
        self.device = device
        self.num_samples = num_samples
        self.delta = delta
        self.n_features = len(feature_names)

    def _get_n_real(self, x_window):
        active_steps = np.where(np.abs(x_window).sum(axis=1) > 0)[0]
        return int(active_steps.max() + 1) if len(active_steps) > 0 else self.window

    def _score(self, x_window, n_real=None):
        if n_real is None:
            n_real = self._get_n_real(x_window)
        with torch.no_grad():
            xb = torch.from_numpy(x_window).float().unsqueeze(0).to(self.device)
            lengths = torch.tensor([n_real], dtype=torch.int64)
            out = self.model(xb, lengths)
            logits = out[0] if isinstance(out, tuple) else out
            if logits.ndim == 3:
                logits = logits[:, n_real - 1, :]
            probability = torch.sigmoid(logits).item()
        return probability

    def capture_relevant_history(self, x_window):
        n_real = self._get_n_real(x_window)
        original_score = self._score(x_window, n_real=n_real)
        effective_len = n_real

        for k in range(1, n_real + 1):
            truncated = x_window.copy()
            number_removed = n_real - k
            if number_removed > 0:
                truncated[:number_removed] = 0.0
            score_k = self._score(truncated, n_real=n_real)
            if abs(score_k - original_score) <= self.delta:
                effective_len = k
                break

        print(f"Score original       : {original_score:.6f}")
        print(f"Longueur utile       : {n_real}/{self.window}")
        print(f"Historique retenu    : {effective_len}/{n_real}")
        print(f"Tolérance delta      : {self.delta}")
        return original_score, effective_len, n_real

    def weighted_sampling(self, x_window, effective_len, n_real):
        samples = np.zeros((self.num_samples, self.window, self.n_features), dtype=np.float32)
        first_relevant = n_real - effective_len

        for s in range(self.num_samples):
            perturbed = np.zeros_like(x_window)
            for t in range(n_real):
                if t < first_relevant:
                    continue
                relative_position = t + 1
                weight = relative_position / (relative_position + 1)
                num_selected = max(1, int(round(weight * self.n_features)))
                num_selected = min(num_selected, self.n_features)
                selected_features = np.random.choice(self.n_features, size=num_selected, replace=False)
                perturbed[t, selected_features] = x_window[t, selected_features]
            samples[s] = perturbed
        return samples

    def sparse_group_lasso(self, x_window, effective_len, n_real):
        import asgl

        samples = self.weighted_sampling(x_window, effective_len, n_real)

        with torch.no_grad():
            xb = torch.from_numpy(samples).float().to(self.device)
            lengths = torch.tensor([n_real] * self.num_samples, dtype=torch.int64)
            out = self.model(xb, lengths)
            logits = out[0] if isinstance(out, tuple) else out
            if logits.ndim == 3:
                logits = logits[:, n_real - 1, :]
            scores = torch.sigmoid(logits).cpu().numpy().reshape(-1)

        X = samples.reshape(self.num_samples, -1)
        y_surrogate = scores.astype(np.float64)
        n_samples = X.shape[0]

        group_index = []
        gid = 1
        for _t in range(self.window):
            for _group_name, features in self.group_dict.items():
                group_index.extend([gid] * len(features))
                gid += 1
        group_index = np.array(group_index, dtype=int)

        rng = np.random.RandomState(42)
        indices = rng.permutation(n_samples)
        split = int(0.75 * n_samples)
        train_idx, val_idx = indices[:split], indices[split:]
        X_train_sur, X_val_sur = X[train_idx], X[val_idx]
        y_train_sur, y_val_sur = y_surrogate[train_idx], y_surrogate[val_idx]

        lambda_grid = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        alpha_grid = [0.1, 0.25, 0.5, 0.9]

        best_error = np.inf
        best_lambda = None
        best_alpha = None

        for lambda1 in lambda_grid:
            for alpha in alpha_grid:
                try:
                    surrogate = asgl.Regressor(
                        model="lm", penalization="sgl", lambda1=lambda1,
                        alpha=alpha, fit_intercept=True, verbose=False,
                    )
                    surrogate.fit(X_train_sur, y_train_sur, group_index=group_index)
                    prediction = np.asarray(surrogate.predict(X_val_sur)).reshape(-1)
                    error = np.mean((prediction - y_val_sur) ** 2)
                    if error < best_error:
                        best_error, best_lambda, best_alpha = error, lambda1, alpha
                except Exception:
                    continue

        if best_lambda is None:
            raise RuntimeError("Aucune configuration ASGL n'a fonctionné.")

        final_model = asgl.Regressor(
            model="lm", penalization="sgl", lambda1=best_lambda,
            alpha=best_alpha, fit_intercept=True, verbose=False,
        )
        final_model.fit(X, y_surrogate, group_index=group_index)

        coef = np.asarray(final_model.coef_).reshape(-1)
        if len(coef) == X.shape[1] + 1:
            coef = coef[1:]

        importance = coef.reshape(self.window, self.n_features)
        return importance, best_lambda, best_alpha, best_error

    def get_top_features(self, importance, top_n=5):
        global_importance = np.sum(np.abs(importance), axis=0)
        ranking = np.argsort(global_importance)[::-1]
        top_indices = ranking[:top_n]
        return [(self.feature_names[idx], float(global_importance[idx])) for idx in top_indices]

    def visualize(self, importance, title="", save_path=None):
        """Optionnel — génère la heatmap matplotlib (non appelée par les
        scripts de pipeline, qui tournent en mode batch/headless)."""
        import matplotlib.pyplot as plt

        vmax = np.max(np.abs(importance))
        if vmax == 0:
            vmax = 1.0
        fig, ax = plt.subplots(figsize=(14, 6))
        image = ax.imshow(importance, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(self.n_features))
        ax.set_xticklabels(self.feature_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(self.window))
        ax.set_yticklabels([f"t-{self.window - 1 - i}" for i in range(self.window)])
        ax.set_xlabel("Features du flux")
        ax.set_ylabel("Historique des flux")
        ax.set_title(title)
        plt.colorbar(image, ax=ax, label="Coefficient Sparse Group Lasso")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    def explain(self, x_window, top_n=12):
        original_score, effective_len, n_real = self.capture_relevant_history(x_window)
        importance, best_lambda, best_alpha, mse = self.sparse_group_lasso(x_window, effective_len, n_real)
        top_features = self.get_top_features(importance, top_n=top_n)
        return {
            "importance": importance,
            "original_score": original_score,
            "effective_history": effective_len,
            "n_real": n_real,
            "lambda1": best_lambda,
            "alpha": best_alpha,
            "surrogate_mse": mse,
            "top_features": top_features,
        }


def build_feature_groups_from_onehot(model_features, categorical_prefixes):
    """Regroupe les colonnes one-hot par famille catégorique (préfixe
    ``<categorie>_``) et les features numériques restantes dans un
    groupe ``trafic``. Utilisé pour Zeek, dont la liste de features
    finale dépend du dataset (``pd.get_dummies``) et n'est donc pas
    connue statiquement — à la différence de CasinoLimit, dont les
    features sont fixes (voir ``casinolimit_data.SEQ_FEATURES`` /
    ``FEATURE_GROUPS``)."""
    groups = {}
    numeric = []
    for col in model_features:
        matched = False
        for cat in categorical_prefixes:
            if col.startswith(f"{cat}_"):
                groups.setdefault(cat, []).append(col)
                matched = True
                break
        if not matched:
            numeric.append(col)
    if numeric:
        groups["trafic"] = numeric
    return groups


def build_top_features_text(top_features):
    """Formate les top features xNIDS en texte lisible par Agent 1, dans
    le même esprit que ``build_explanation_text`` du pipeline SHAP
    (``src/ids_pipeline/explainability``)."""
    lines = []
    for rank, (feature, score) in enumerate(top_features, start=1):
        lines.append(f"Rank {rank} | importance={score:.6f} | {feature}")
    return "\n".join(lines)
