import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# ── Helpers ──────────────────────────────────────────────────────────

def draw_box(ax, cx, cy, w, h, title, lines, border_color, fill_color):
    """Draw a rounded box centered at (cx, cy) with title + body lines."""
    x0, y0 = cx - w / 2, cy - h / 2
    patch = FancyBboxPatch((x0, y0), w, h,
                           boxstyle='round,pad=0.08',
                           facecolor=fill_color,
                           edgecolor=border_color,
                           linewidth=2.0)
    ax.add_patch(patch)

    # Title bar divider
    ax.plot([x0 + 0.12, x0 + w - 0.12], [cy + h/2 - 0.44, cy + h/2 - 0.44],
            color=border_color, lw=0.8, alpha=0.5)

    # Title
    ax.text(cx, cy + h/2 - 0.22, title,
            ha='center', va='center',
            fontsize=11, fontweight='bold', color=border_color)

    # Body lines — evenly spaced
    n = len(lines)
    if n == 0:
        return
    body_h = h - 0.55
    spacing = body_h / (n + 1)
    for i, line in enumerate(lines):
        y = (cy - h/2) + body_h - (i * spacing)
        ax.text(cx, y, line,
                ha='center', va='center',
                fontsize=9, color='#333333')


def arrow(ax, x1, y1, x2, y2, label='', color='#555555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=2.0, mutation_scale=16))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.18
        ax.text(mx, my, label, ha='center', va='bottom',
                fontsize=8, color=color)


# ── Title ────────────────────────────────────────────────────────────
ax.text(8, 8.55, 'GAT-LSTM Architecture  —  Wildfire Risk Prediction',
        ha='center', va='center',
        fontsize=15, fontweight='bold', color='#111111')
ax.text(8, 8.12, '100 sensor nodes   ×   6 time steps   ×   25 features per node',
        ha='center', va='center',
        fontsize=10, color='#666666')

# ── Main flow boxes (y = 5.6) ────────────────────────────────────────
Y = 5.6

# 1. Input
draw_box(ax, cx=1.6, cy=Y, w=2.4, h=3.2,
         title='INPUT',
         lines=[
             '100 nodes',
             '6 time steps',
             '25 features',
             '──────────',
             'temperature',
             'humidity',
             'surface temp',
             'smoke index',
             'CO  /  VOC',
             'wind speed',
         ],
         border_color='#1565c0',
         fill_color='#e3f2fd')

# 2. GAT Layer
draw_box(ax, cx=5.2, cy=Y, w=2.6, h=3.2,
         title='GAT LAYER',
         lines=[
             'Spatial Processing',
             '──────────',
             '4 attention heads',
             'Nodes within 1.5 km',
             'share information',
             '──────────',
             'Output: 100 × 64',
             '(spatial embedding)',
         ],
         border_color='#b71c1c',
         fill_color='#ffebee')

# 3. LSTM Layer
draw_box(ax, cx=9.0, cy=Y, w=2.6, h=3.2,
         title='LSTM LAYER',
         lines=[
             'Temporal Processing',
             '──────────',
             '6 time steps',
             'Learns trends over',
             'last 3 minutes',
             '──────────',
             'Output: 100 × 128',
             '(temporal pattern)',
         ],
         border_color='#e65100',
         fill_color='#fff3e0')

# 4. Linear Head
draw_box(ax, cx=12.4, cy=Y, w=2.2, h=3.2,
         title='LINEAR HEAD',
         lines=[
             'Fully Connected',
             '──────────',
             '128  →  64  →  1',
             'Sigmoid activation',
             '──────────',
             'Output: 100 × 1',
             '(one score per node)',
         ],
         border_color='#2e7d32',
         fill_color='#e8f5e9')

# 5. Output
draw_box(ax, cx=15.0, cy=Y, w=1.6, h=3.2,
         title='OUTPUT',
         lines=[
             'Fire Risk',
             'Score',
             '──────────',
             '0.0 – 1.0',
             'per node',
             '──────────',
             'Threshold',
             '≥ 0.80',
         ],
         border_color='#e65100',
         fill_color='#fff8e1')

# ── Arrows between boxes ─────────────────────────────────────────────
arrow(ax, 2.82, Y, 3.88, Y, label='25 features\nper step', color='#1565c0')
arrow(ax, 6.52, Y, 7.68, Y, label='64-dim\nspatial', color='#b71c1c')
arrow(ax, 10.32, Y, 11.28, Y, label='128-dim\ntemporal', color='#e65100')
arrow(ax, 13.52, Y, 14.18, Y, color='#2e7d32')

# ── Time step section ─────────────────────────────────────────────────
ax.plot([0.3, 15.7], [3.38, 3.38], '-', color='#dddddd', lw=1.2)
ax.text(8, 3.1, 'GAT runs independently on each time step  →  6 spatial outputs stacked  →  LSTM reads the sequence',
        ha='center', va='center', fontsize=9, color='#888888', style='italic')

# Time step boxes
step_labels = ['t − 5\n(−2.5 min)', 't − 4\n(−2.0 min)', 't − 3\n(−1.5 min)',
               't − 2\n(−1.0 min)', 't − 1\n(−0.5 min)', 't\n(now)']
step_colors = ['#e3f2fd', '#e3f2fd', '#e3f2fd', '#e3f2fd', '#e3f2fd', '#bbdefb']

for i, (lbl, col) in enumerate(zip(step_labels, step_colors)):
    bx = 1.3 + i * 2.42
    patch = FancyBboxPatch((bx - 0.9, 1.62), 1.78, 1.12,
                           boxstyle='round,pad=0.06',
                           facecolor=col,
                           edgecolor='#1565c0',
                           linewidth=1.3)
    ax.add_patch(patch)
    ax.text(bx, 2.18, lbl,
            ha='center', va='center',
            fontsize=8.5, color='#1565c0')
    if i < 5:
        arrow(ax, bx + 0.9, 2.18, bx + 1.54, 2.18, color='#aaaaaa')

# ── Legend ────────────────────────────────────────────────────────────
ax.plot([0.3, 15.7], [1.22, 1.22], '-', color='#dddddd', lw=1.0)

legend_items = [
    ('#1565c0', '#e3f2fd', 'Input  —  Raw sensor features'),
    ('#b71c1c', '#ffebee', 'GAT  —  Spatial graph attention'),
    ('#e65100', '#fff3e0', 'LSTM  —  Temporal sequence memory'),
    ('#2e7d32', '#e8f5e9', 'Linear Head  —  Risk score regression'),
    ('#e65100', '#fff8e1', 'Output  —  Fire risk per node (0–1)'),
]

for i, (ec, fc, lbl) in enumerate(legend_items):
    lx = 0.5 + i * 3.1
    patch = FancyBboxPatch((lx, 0.55), 0.32, 0.32,
                           boxstyle='round,pad=0.03',
                           facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(patch)
    ax.text(lx + 0.44, 0.71, lbl,
            va='center', fontsize=8.5, color='#333333')

plt.tight_layout(pad=0.4)
plt.savefig('GAT_LSTM_Architecture.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('Saved: GAT_LSTM_Architecture.png')
