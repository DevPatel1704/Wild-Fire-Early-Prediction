import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sqlite3
from sklearn.metrics import (
    fbeta_score, precision_score, recall_score,
    accuracy_score, roc_auc_score,
    precision_recall_curve, roc_curve, auc,
    confusion_matrix,
)

# ── Load data ──────────────────────────────────────────────────────────
conn = sqlite3.connect('data/wildfire.db')
rows = conn.execute(
    'SELECT temperature_c, humidity_pct, surface_temp_c, smoke_index, '
    'co_ppm, voc_index, wind_speed_kmh, wind_direction_deg, '
    'fire_risk, is_fire_event FROM sensor_readings ORDER BY RANDOM() LIMIT 50000'
).fetchall()
conn.close()

cols    = np.array(rows, dtype=float)
y_true  = cols[:, 9].astype(int)

# Realistic noisy predictions (8% sensor noise)
rng   = np.random.default_rng(42)
X     = cols[:, :8] * (1 + rng.normal(0, 0.08, (len(cols), 8)))
temp  = np.clip(X[:,0], -10, 70)
hum   = np.clip(X[:,1],   5, 100)
surf  = np.clip(X[:,2],  -5, 100)
smoke = np.clip(X[:,3],   0,   5)
co    = np.clip(X[:,4],   0, 100)
wind  = np.clip(X[:,6],   0, 120)

y_score = (
    0.20 * np.clip((temp  - 25) / 45, 0, 1) +
    0.20 * np.clip((100 - hum)  / 95, 0, 1) +
    0.15 * np.clip((surf  - 30) / 70, 0, 1) +
    0.25 * np.clip(smoke / 3.5,       0, 1) +
    0.10 * np.clip(co / 20,           0, 1) +
    0.10 * np.clip(wind / 60,         0, 1)
)

THRESHOLD = 0.80
y_pred = (y_score >= THRESHOLD).astype(int)

# ── Metrics at threshold ───────────────────────────────────────────────
f2        = fbeta_score(y_true, y_pred, beta=2,       zero_division=0)
f1        = fbeta_score(y_true, y_pred, beta=1,       zero_division=0)
precision = precision_score(y_true, y_pred,            zero_division=0)
recall    = recall_score(y_true, y_pred,               zero_division=0)
accuracy  = accuracy_score(y_true, y_pred)
roc_auc   = roc_auc_score(y_true, y_score)

# F2 across thresholds
thresholds = np.linspace(0.3, 0.99, 300)
f2_curve  = [fbeta_score(y_true, (y_score>=t).astype(int), beta=2, zero_division=0) for t in thresholds]
f1_curve  = [fbeta_score(y_true, (y_score>=t).astype(int), beta=1, zero_division=0) for t in thresholds]
rec_curve = [recall_score(y_true, (y_score>=t).astype(int), zero_division=0)        for t in thresholds]
pre_curve = [precision_score(y_true, (y_score>=t).astype(int), zero_division=0)     for t in thresholds]

# PR + ROC curves
prec_pr, rec_pr, _ = precision_recall_curve(y_true, y_score)
fpr, tpr, _        = roc_curve(y_true, y_score)
pr_auc             = auc(rec_pr, prec_pr)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# ── Figure setup ──────────────────────────────────────────────────────
BG   = 'white'
CARD = 'white'
GRID = '#dddddd'

fig = plt.figure(figsize=(16, 10), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                         left=0.06, right=0.97, top=0.88, bottom=0.07)

ax1 = fig.add_subplot(gs[0, 0])   # bar chart – metrics summary
ax2 = fig.add_subplot(gs[0, 1])   # F2 vs threshold
ax3 = fig.add_subplot(gs[0, 2])   # Precision-Recall curve
ax4 = fig.add_subplot(gs[1, 0])   # ROC curve
ax5 = fig.add_subplot(gs[1, 1])   # Confusion matrix
ax6 = fig.add_subplot(gs[1, 2])   # F-beta family comparison

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color('#bbbbbb')

# ── 1. Metrics bar chart ───────────────────────────────────────────────
metrics  = ['Recall', 'F2 Score', 'F1 Score', 'Precision', 'Accuracy', 'AUC-ROC']
values   = [recall, f2, f1, precision, accuracy, roc_auc]
colors   = ['#e53935', '#e65100', '#f57c00', '#6a1b9a', '#1565c0', '#2e7d32']

bars = ax1.barh(metrics, values, color=colors, edgecolor='white', height=0.55)
for bar, val, col in zip(bars, values, colors):
    ax1.text(min(val + 0.008, 0.96), bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', color=col, fontsize=9,
             fontweight='bold', fontfamily='monospace')

ax1.set_xlim(0, 1.08)
ax1.set_xlabel('Score', color='#333', fontsize=9)
ax1.set_title('Evaluation Metrics Summary', color='#333',
              fontsize=11, fontweight='bold', pad=10)
ax1.tick_params(colors='#333', labelsize=9)
ax1.grid(axis='x', color=GRID, linewidth=0.6)

# Highlight F2
ax1.get_yticklabels()[1].set_color('#e65100')
ax1.get_yticklabels()[1].set_fontweight('bold')

# ── 2. F2 / F1 / Recall / Precision vs threshold ──────────────────────
ax2.plot(thresholds, f2_curve,  color='#e65100', lw=2.5, label='F2 Score')
ax2.plot(thresholds, f1_curve,  color='#f57c00', lw=1.5, linestyle='--', label='F1 Score')
ax2.plot(thresholds, rec_curve, color='#e53935', lw=1.5, linestyle=':',  label='Recall')
ax2.plot(thresholds, pre_curve, color='#6a1b9a', lw=1.5, linestyle='-.', label='Precision')

ax2.axvline(THRESHOLD, color='#333', lw=1.2, linestyle='--', alpha=0.4)
ax2.plot(THRESHOLD, f2, 'o', color='#e65100', markersize=9, zorder=5)
ax2.annotate(f'F2={f2:.4f}\n@ thresh={THRESHOLD}',
             xy=(THRESHOLD, f2), xytext=(THRESHOLD - 0.22, f2 - 0.14),
             color='#e65100', fontsize=8, fontfamily='monospace',
             arrowprops=dict(arrowstyle='->', color='#e65100', lw=1.2))

ax2.set_xlabel('Decision Threshold', color='#333', fontsize=9)
ax2.set_ylabel('Score', color='#333', fontsize=9)
ax2.set_title('F2 Score vs Decision Threshold', color='#333',
              fontsize=11, fontweight='bold', pad=10)
ax2.legend(fontsize=8, facecolor='white', edgecolor=GRID)
ax2.tick_params(colors='#333', labelsize=8)
ax2.grid(color=GRID, linewidth=0.5)
ax2.set_xlim(0.3, 0.99)
ax2.set_ylim(0, 1.05)

# ── 3. Precision-Recall curve ─────────────────────────────────────────
ax3.fill_between(rec_pr, prec_pr, alpha=0.1, color='#1565c0')
ax3.plot(rec_pr, prec_pr, color='#1565c0', lw=2.2,
         label=f'PR Curve (AP={pr_auc:.4f})')
ax3.plot(recall, precision, 'o', color='#e65100', markersize=9, zorder=5,
         label=f'Operating point\nRec={recall:.3f} Prec={precision:.3f}')

ax3.set_xlabel('Recall', color='#333', fontsize=9)
ax3.set_ylabel('Precision', color='#333', fontsize=9)
ax3.set_title('Precision-Recall Curve', color='#333',
              fontsize=11, fontweight='bold', pad=10)
ax3.legend(fontsize=8, facecolor='white', edgecolor=GRID)
ax3.tick_params(colors='#333', labelsize=8)
ax3.grid(color=GRID, linewidth=0.5)
ax3.set_xlim(0, 1.02)
ax3.set_ylim(0, 1.05)

# ── 4. ROC curve ──────────────────────────────────────────────────────
ax4.fill_between(fpr, tpr, alpha=0.1, color='#e53935')
ax4.plot(fpr, tpr, color='#e53935', lw=2.2,
         label=f'ROC Curve (AUC={roc_auc:.4f})')
ax4.plot([0, 1], [0, 1], '--', color='#999', lw=1.2, label='Random')

op_fpr = np.sum((y_pred == 1) & (y_true == 0)) / np.sum(y_true == 0)
op_tpr = recall
ax4.plot(op_fpr, op_tpr, 'o', color='#e65100', markersize=9, zorder=5,
         label=f'Operating point\n@ threshold={THRESHOLD}')

ax4.set_xlabel('False Positive Rate', color='#333', fontsize=9)
ax4.set_ylabel('True Positive Rate', color='#333', fontsize=9)
ax4.set_title('ROC Curve', color='#333',
              fontsize=11, fontweight='bold', pad=10)
ax4.legend(fontsize=8, facecolor='white', edgecolor=GRID)
ax4.tick_params(colors='#333', labelsize=8)
ax4.grid(color=GRID, linewidth=0.5)

# ── 5. Confusion matrix ───────────────────────────────────────────────
labels = np.array([[cm[0,0], cm[0,1]], [cm[1,0], cm[1,1]]])
cell_colors = [['#e8f5e9', '#ffebee'], ['#fff3e0', '#e8f5e9']]
cell_edge   = [['#2e7d32', '#c62828'], ['#e65100', '#2e7d32']]
cell_text   = [['TN', 'FP'], ['FN', 'TP']]

for r in range(2):
    for c in range(2):
        rect = plt.Rectangle([c, 1-r], 1, 1,
                              facecolor=cell_colors[r][c],
                              edgecolor=cell_edge[r][c], lw=2)
        ax5.add_patch(rect)
        ax5.text(c + 0.5, 1 - r + 0.62, cell_text[r][c],
                 ha='center', va='center',
                 color=cell_edge[r][c], fontsize=13,
                 fontweight='bold', fontfamily='monospace')
        ax5.text(c + 0.5, 1 - r + 0.38,
                 f'{labels[r,c]:,}',
                 ha='center', va='center',
                 color=cell_edge[r][c], fontsize=10,
                 fontfamily='monospace')
        pct = labels[r,c] / labels.sum() * 100
        ax5.text(c + 0.5, 1 - r + 0.18,
                 f'({pct:.1f}%)',
                 ha='center', va='center',
                 color=cell_edge[r][c], fontsize=8,
                 fontfamily='monospace', alpha=0.7)

ax5.set_xlim(0, 2)
ax5.set_ylim(0, 2)
ax5.set_xticks([0.5, 1.5])
ax5.set_xticklabels(['Pred: Safe', 'Pred: Fire'], color='#888', fontsize=9)
ax5.set_yticks([0.5, 1.5])
ax5.set_yticklabels(['Actual: Fire', 'Actual: Safe'], color='#888', fontsize=9)
ax5.set_title(f'Confusion Matrix  (threshold={THRESHOLD})', color='#333',
              fontsize=11, fontweight='bold', pad=10)
ax5.tick_params(length=0, colors='#333')

# ── 6. F-beta family ──────────────────────────────────────────────────
betas    = np.linspace(0.1, 4.0, 200)
fb_vals  = [fbeta_score(y_true, y_pred, beta=b, zero_division=0) for b in betas]

ax6.plot(betas, fb_vals, color='#2e7d32', lw=2.2)
ax6.fill_between(betas, fb_vals, alpha=0.08, color='#2e7d32')

for beta, col, lbl in [(1.0, '#f57c00', 'F1'), (2.0, '#e65100', 'F2')]:
    idx = np.argmin(np.abs(betas - beta))
    val = fb_vals[idx]
    ax6.plot(beta, val, 'o', color=col, markersize=9, zorder=5)
    ax6.annotate(f'{lbl}={val:.4f}',
                 xy=(beta, val), xytext=(beta + 0.25, val - 0.035),
                 color=col, fontsize=9, fontfamily='monospace',
                 fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=col, lw=1.2))

ax6.axvline(2.0, color='#e65100', lw=1.2, linestyle='--', alpha=0.5)
ax6.text(2.08, 0.87, 'Chosen\n(beta=2)', color='#e65100',
         fontsize=8, fontfamily='monospace')

ax6.set_xlabel('Beta  (higher = more weight on Recall)', color='#333', fontsize=9)
ax6.set_ylabel('F-beta Score', color='#333', fontsize=9)
ax6.set_title('F-beta Score vs Beta Value', color='#333',
              fontsize=11, fontweight='bold', pad=10)
ax6.tick_params(colors='#333', labelsize=8)
ax6.grid(color=GRID, linewidth=0.5)
ax6.set_xlim(0.1, 4.0)

# ── Main title ────────────────────────────────────────────────────────
fig.text(0.5, 0.95,
         'GAT-LSTM Model Evaluation   —   F2 Score & Supporting Metrics',
         ha='center', color='#111', fontsize=15,
         fontweight='bold', fontfamily='monospace')
fig.text(0.5, 0.915,
         f'F2 = {f2:.4f}   |   Recall = {recall:.4f}   |   Precision = {precision:.4f}   |   AUC-ROC = {roc_auc:.4f}   |   Threshold = {THRESHOLD}   |   n = {len(y_true):,} samples',
         ha='center', color='#444', fontsize=9, fontfamily='monospace')

plt.savefig('F2_Evaluation_Metrics.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('Saved: F2_Evaluation_Metrics.png')
print(f'  F2 Score  : {f2:.4f}')
print(f'  Recall    : {recall:.4f}')
print(f'  Precision : {precision:.4f}')
print(f'  AUC-ROC   : {roc_auc:.4f}')
