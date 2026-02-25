"""
Logistic Regression vs Random Forest vs SVM (Linear Kernel) — 3-Model Comparison
==================================================================================
All 3 models use identical feature extraction for a fair comparison:
  - TF-IDF trigram (1-3), max 5000 features, sublinear TF scaling
  - + similarity_score numeric feature (standardized)

Requirements:
    pip install scikit-learn pandas numpy matplotlib seaborn scipy

Usage:
    python ml_comparison.py

Make sure your CSV file path is set correctly in the CONFIG section below.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)

# ── CONFIG — update these as needed ──────────────────────────────────────────
CSV_PATH       = "final_merged_dataset.csv"   # path to your CSV file
TEXT_COLUMN    = "text"                        # name of the text column
NUMERIC_COLUMN = "similarity_score"           # name of the numeric feature column
LABEL_COLUMN   = "label"                      # name of the label column
OUTPUT_PLOT    = "model_comparison.png"        # output plot filename
# ─────────────────────────────────────────────────────────────────────────────

# ── Load & prep ───────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
df[TEXT_COLUMN]    = df[TEXT_COLUMN].fillna('')
df[NUMERIC_COLUMN] = df[NUMERIC_COLUMN].fillna(df[NUMERIC_COLUMN].median())

print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df[LABEL_COLUMN].value_counts()}\n")

X_text = df[TEXT_COLUMN]
X_num  = df[[NUMERIC_COLUMN]].values
y      = df[LABEL_COLUMN]

# Train/test split (stratified 80/20)
X_text_tr, X_text_te, X_num_tr, X_num_te, y_tr, y_te = train_test_split(
    X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
)

# ── Shared feature extraction (same for ALL 3 models) ────────────────────────
print("Fitting trigram TF-IDF (shared across all models)...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), sublinear_tf=True)
X_tfidf_tr = tfidf.fit_transform(X_text_tr)
X_tfidf_te = tfidf.transform(X_text_te)

scaler = StandardScaler()
X_num_tr_sc = scaler.fit_transform(X_num_tr)
X_num_te_sc = scaler.transform(X_num_te)

# Single combined feature matrix used by all models
X_tr = hstack([X_tfidf_tr, csr_matrix(X_num_tr_sc)])
X_te = hstack([X_tfidf_te, csr_matrix(X_num_te_sc)])

# ── Train models (same features, different algorithms) ────────────────────────
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr.fit(X_tr, y_tr)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

# LinearSVC doesn't output probabilities natively — wrap with CalibratedClassifierCV
print("Training SVM (Linear Kernel)...")
svm_base = LinearSVC(C=1.0, max_iter=2000, random_state=42)
svm = CalibratedClassifierCV(svm_base, cv=5)
svm.fit(X_tr, y_tr)

# ── Predictions ───────────────────────────────────────────────────────────────
lr_pred   = lr.predict(X_te)
rf_pred   = rf.predict(X_te)
svm_pred  = svm.predict(X_te)

lr_proba  = lr.predict_proba(X_te)[:, 1]
rf_proba  = rf.predict_proba(X_te)[:, 1]
svm_proba = svm.predict_proba(X_te)[:, 1]

# ── 5-Fold Cross Validation ───────────────────────────────────────────────────
print("Running 5-fold cross validation (this may take a moment)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_full = hstack([tfidf.transform(X_text), csr_matrix(scaler.transform(X_num))])

lr_cv_acc  = cross_val_score(lr,  X_full, y, cv=cv, scoring='accuracy')
rf_cv_acc  = cross_val_score(rf,  X_full, y, cv=cv, scoring='accuracy')
svm_cv_acc = cross_val_score(svm, X_full, y, cv=cv, scoring='accuracy')

lr_cv_auc  = cross_val_score(lr,  X_full, y, cv=cv, scoring='roc_auc')
rf_cv_auc  = cross_val_score(rf,  X_full, y, cv=cv, scoring='roc_auc')
svm_cv_auc = cross_val_score(svm, X_full, y, cv=cv, scoring='roc_auc')

# ── Print summary metrics ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("  RESULTS SUMMARY")
print("="*70)
metrics = {
    'Model':          ['Logistic Regression', 'Random Forest', 'SVM (Linear)'],
    'Test Accuracy':  [accuracy_score(y_te, lr_pred),  accuracy_score(y_te, rf_pred),  accuracy_score(y_te, svm_pred)],
    'ROC-AUC':        [roc_auc_score(y_te, lr_proba),  roc_auc_score(y_te, rf_proba),  roc_auc_score(y_te, svm_proba)],
    'Avg Precision':  [average_precision_score(y_te, lr_proba), average_precision_score(y_te, rf_proba), average_precision_score(y_te, svm_proba)],
    'CV Acc (mean)':  [lr_cv_acc.mean(),  rf_cv_acc.mean(),  svm_cv_acc.mean()],
    'CV Acc (std)':   [lr_cv_acc.std(),   rf_cv_acc.std(),   svm_cv_acc.std()],
    'CV AUC (mean)':  [lr_cv_auc.mean(),  rf_cv_auc.mean(),  svm_cv_auc.mean()],
    'CV AUC (std)':   [lr_cv_auc.std(),   rf_cv_auc.std(),   svm_cv_auc.std()],
}
mdf = pd.DataFrame(metrics)
print(mdf.to_string(index=False))

MODELS = {
    'Logistic Regression': (lr_pred,  lr_proba,  '#4fc3f7'),
    'Random Forest':       (rf_pred,  rf_proba,  '#ef9a9a'),
    'SVM (Linear Trigram)':(svm_pred, svm_proba, '#a5d6a7'),
}

for name, (pred, proba, _) in MODELS.items():
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    print(classification_report(y_te, pred, target_names=['Class 0', 'Class 1']))

# ── Plot ──────────────────────────────────────────────────────────────────────
print(f"\nGenerating plots -> {OUTPUT_PLOT}")
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#0f1117')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

COLORS = {
    'Logistic Regression':  '#4fc3f7',
    'Random Forest':        '#ef9a9a',
    'SVM (Linear Trigram)': '#a5d6a7',
}

def ax_style(ax, title):
    ax.set_facecolor('#1a1d27')
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=10)
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')
    ax.xaxis.label.set_color('#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')

# 1. ROC Curve
ax1 = fig.add_subplot(gs[0, 0])
for name, (pred, proba, col) in MODELS.items():
    fpr, tpr, _ = roc_curve(y_te, proba)
    auc = roc_auc_score(y_te, proba)
    ax1.plot(fpr, tpr, color=col, lw=2, label=f'{name} (AUC={auc:.3f})')
ax1.plot([0, 1], [0, 1], '--', color='#555566', lw=1)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(fontsize=7.5, facecolor='#1a1d27', labelcolor='white', framealpha=0.8)
ax_style(ax1, 'ROC Curve')

# 2. Precision-Recall Curve
ax2 = fig.add_subplot(gs[0, 1])
for name, (pred, proba, col) in MODELS.items():
    prec, rec, _ = precision_recall_curve(y_te, proba)
    ap = average_precision_score(y_te, proba)
    ax2.plot(rec, prec, color=col, lw=2, label=f'{name} (AP={ap:.3f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.legend(fontsize=7.5, facecolor='#1a1d27', labelcolor='white', framealpha=0.8)
ax_style(ax2, 'Precision-Recall Curve')

# 3. CV Accuracy Boxplot
ax3 = fig.add_subplot(gs[0, 2])
cv_data = [lr_cv_acc, rf_cv_acc, svm_cv_acc]
bp = ax3.boxplot(cv_data, patch_artist=True, widths=0.4,
                 medianprops=dict(color='white', lw=2))
box_colors = [COLORS['Logistic Regression'], COLORS['Random Forest'], COLORS['SVM (Linear Trigram)']]
for patch, col in zip(bp['boxes'], box_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
for whisker in bp['whiskers']: whisker.set_color('#aaaaaa')
for cap in bp['caps']:         cap.set_color('#aaaaaa')
for flier in bp['fliers']:     flier.set_markerfacecolor('#aaaaaa')
ax3.set_xticks([1, 2, 3])
ax3.set_xticklabels(['Logistic\nRegression', 'Random\nForest', 'SVM\n(Linear)'], color='white', fontsize=9)
ax3.set_ylabel('Accuracy')
ax_style(ax3, '5-Fold CV Accuracy')

# 4. Confusion Matrices (one per model)
for i, (name, (pred, proba, col)) in enumerate(MODELS.items()):
    ax = fig.add_subplot(gs[1, i])
    cm = confusion_matrix(y_te, pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                linewidths=0.5, linecolor='#333344',
                annot_kws={'size': 14, 'color': 'white'},
                cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax_style(ax, f'Confusion Matrix\n{name}')
    ax.tick_params(colors='white')

# ── Overlay: Key Metrics Bar Chart (replaces 3rd confusion matrix slot) ───────
# Re-draw last subplot as metrics bar chart instead
fig.delaxes(fig.axes[-1])   # remove the 3rd confusion matrix
ax5 = fig.add_subplot(gs[1, 2])
metric_names = ['Test Accuracy', 'ROC-AUC', 'Avg Precision']
model_keys   = ['Logistic Regression', 'Random Forest', 'SVM (Linear Trigram)']
mdf_vals = {
    'Logistic Regression':  [mdf.loc[0, m] for m in metric_names],
    'Random Forest':        [mdf.loc[1, m] for m in metric_names],
    'SVM (Linear Trigram)': [mdf.loc[2, m] for m in metric_names],
}
x = np.arange(len(metric_names))
w = 0.22
offsets = [-w, 0, w]
for (mname, vals), offset in zip(mdf_vals.items(), offsets):
    bars = ax5.bar(x + offset, vals, w, color=COLORS[mname], alpha=0.85, label=mname)
    for bar, v in zip(bars, vals):
        ax5.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                 f'{v:.3f}', ha='center', va='bottom', color='white', fontsize=7)
ax5.set_xticks(x)
ax5.set_xticklabels(metric_names, rotation=12, ha='right')
ax5.set_ylim(0, 1.15)
ax5.legend(fontsize=7, facecolor='#1a1d27', labelcolor='white', framealpha=0.8)
ax_style(ax5, 'Key Metrics Comparison')

fig.suptitle('Logistic Regression vs Random Forest vs SVM (Linear) — 3-Model Comparison',
             color='white', fontsize=14, fontweight='bold', y=0.98)

plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Plot saved to: {OUTPUT_PLOT}")
plt.show()
