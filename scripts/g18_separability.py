"""Gate 18 Part 2: Test separability of latent features between poisoned and clean episodes."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Load data
with open("./runs/g18_latent_features.json") as f:
    data = json.load(f)

clean = data["clean"]
poisoned = data["poisoned"]

# Extract features
feature_names = ["latent_norm_mean", "latent_norm_max", "norm_ratio_mean", "norm_ratio_max",
                 "passage_divergence_mean", "passage_divergence_max", "n_aug_points"]

# Build feature matrix
# Label: 0 = clean, 1 = poisoned
X_clean = []
X_poison = []

for r in clean:
    f = r["latent_features"]
    if f["n_aug_points"] > 0:
        X_clean.append([f[name] for name in feature_names])

for r in poisoned:
    f = r["latent_features"]
    if f["n_aug_points"] > 0:
        X_poison.append([f[name] for name in feature_names])

X_clean = np.array(X_clean)
X_poison = np.array(X_poison)

n_clean = len(X_clean)
n_poison = len(X_poison)
print(f"Clean episodes with latents: {n_clean}")
print(f"Poisoned episodes with latents: {n_poison}")

# Also extract target_hit for the poisoned set (to separate "poison worked" from "poison failed")
target_hits = np.array([1 if r.get("target_hit") else 0 for r in poisoned if r["latent_features"]["n_aug_points"] > 0])

X = np.vstack([X_clean, X_poison])
y = np.hstack([np.zeros(n_clean), np.ones(n_poison)])

print(f"\n{'='*80}")
print("PART 2: Per-Feature Separability")
print(f"{'='*80}")

# Per-feature AUC
print(f"\n{'Feature':<25} {'Clean mean±std':>20} {'Poison mean±std':>20} {'AUC':>8}")
print("-" * 80)

aucs = {}
for i, name in enumerate(feature_names):
    clean_vals = X_clean[:, i]
    poison_vals = X_poison[:, i]
    
    # AUC (treating feature as a score, higher = more likely poisoned)
    auc = roc_auc_score(y, X[:, i])
    # Also try inverted (lower = more likely poisoned)
    auc_inv = roc_auc_score(y, -X[:, i])
    best_auc = max(auc, auc_inv)
    aucs[name] = best_auc
    
    print(f"{name:<25} {np.mean(clean_vals):>10.4f}±{np.std(clean_vals):>6.4f} "
          f"{np.mean(poison_vals):>10.4f}±{np.std(poison_vals):>6.4f} {best_auc:>8.4f}")

# Fixed-5% FPR detection rate per feature
print(f"\n{'='*80}")
print("Fixed-5% FPR Detection Rate (per feature)")
print(f"{'='*80}")

# To set threshold at 5% FPR on clean:
# Find the 95th percentile of clean values (if higher = poisoned)
# or 5th percentile (if lower = poisoned)

print(f"\n{'Feature':<25} {'Direction':>10} {'Threshold':>12} {'Detection':>10} {'FPR':>8}")
print("-" * 70)

for i, name in enumerate(feature_names):
    auc = aucs[name]
    if auc < 0.5:
        continue  # Skip features with no signal
    
    # Determine direction
    auc_normal = roc_auc_score(y, X[:, i])
    if auc_normal >= 0.5:
        direction = "higher"
        threshold = np.percentile(X_clean[:, i], 95)
        detection = np.mean(X_poison[:, i] > threshold)
    else:
        direction = "lower"
        threshold = np.percentile(X_clean[:, i], 5)
        detection = np.mean(X_poison[:, i] < threshold)
    
    fpr = 0.05  # By construction
    print(f"{name:<25} {direction:>10} {threshold:>12.4f} {detection:>10.1%} {fpr:>8.1%}")

# Also check: among poisoned episodes where the attack SUCCEEDED (target_hit=True),
# is there any signal?
print(f"\n{'='*80}")
print("Among poison-success episodes (target_hit=True) vs clean")
print(f"{'='*80}")

success_mask = target_hits == 1
fail_mask = target_hits == 0
print(f"Poison successes: {success_mask.sum()}/50")
print(f"Poison failures: {fail_mask.sum()}/50")

if success_mask.sum() > 0:
    X_success = X_poison[success_mask]
    y_success = np.hstack([np.zeros(n_clean), np.ones(len(X_success))])
    X_success_full = np.vstack([X_clean, X_success])
    
    print(f"\n{'Feature':<25} {'Clean mean':>12} {'Success mean':>12} {'AUC':>8}")
    print("-" * 65)
    for i, name in enumerate(feature_names):
        auc = roc_auc_score(y_success, X_success_full[:, i])
        auc_inv = roc_auc_score(y_success, -X_success_full[:, i])
        best_auc = max(auc, auc_inv)
        print(f"{name:<25} {np.mean(X_clean[:, i]):>12.4f} {np.mean(X_success[:, i]):>12.4f} {best_auc:>8.4f}")

# Combined features (logistic regression with 5-fold CV)
print(f"\n{'='*80}")
print("Combined Features (Logistic Regression, 5-fold CV)")
print(f"{'='*80}")

# Use top 3 features by AUC
sorted_features = sorted(aucs.items(), key=lambda x: x[1], reverse=True)
top_features = [name for name, auc in sorted_features[:3]]
top_indices = [feature_names.index(name) for name in top_features]

print(f"Top 3 features: {top_features}")
print(f"AUCs: {[aucs[name] for name in top_features]}")

X_top = X[:, top_indices]

# 5-fold CV
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = []
cv_detection_5fpr = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_top, y)):
    X_train, X_test = X_top[train_idx], X_top[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    scores = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, scores)
    cv_aucs.append(auc)
    
    # Fixed 5% FPR on clean test items
    clean_test = scores[y_test == 0]
    poison_test = scores[y_test == 1]
    
    if len(clean_test) > 0 and len(poison_test) > 0:
        threshold = np.percentile(clean_test, 95)
        detection = np.mean(poison_test > threshold)
        cv_detection_5fpr.append(detection)
    
    print(f"  Fold {fold+1}: AUC={auc:.4f}, Detection@5%FPR={detection:.1%}")

print(f"\nMean CV AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
print(f"Mean Detection@5%FPR: {np.mean(cv_detection_5fpr):.1%} ± {np.std(cv_detection_5fpr):.1%}")

# Comparison with failed defenses
print(f"\n{'='*80}")
print("Comparison with Failed Answer-Level Defenses")
print(f"{'='*80}")
print(f"{'Defense':<35} {'Detection':>10} {'FPR':>10} {'Verdict':>15}")
print("-" * 75)
print(f"{'Retrieval-consistency (G15)':<35} {'95.0%':>10} {'98.0%':>10} {'Useless':>15}")
print(f"{'Ablation-disagree tagged (G16)':<35} {'30.0%':>10} {'52.0%':>10} {'Poor':>15}")
print(f"{'Ablation-disagree tagless (G17)':<35} {'96.0%':>10} {'88.0%':>10} {'Poor':>15}")
print(f"{'Latent norm (single, best)':<35} {'—':>10} {'5.0%':>10} {'See above':>15}")
print(f"{'Logistic combo (top-3, CV)':<35} {f'{np.mean(cv_detection_5fpr):.1%}':>10} {'5.0%':>10} {'See above':>15}")
