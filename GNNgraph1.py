import matplotlib.pyplot as plt

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Function to draw boxes
def draw_box(ax, x, y, text, width=0.15, height=0.1, color="lightblue"):
    ax.add_patch(plt.Rectangle((x, y), width, height, edgecolor="black", facecolor=color, lw=1.5))
    ax.text(x + width / 2, y + height / 2, text, fontsize=10, ha="center", va="center", wrap=True)

# Draw arrows for connections first (underneath the boxes)
connections = [
    (0.25, 0.55, 0.3, 0.55),  # Input to Preprocess
    (0.4, 0.55, 0.5, 0.65),  # Preprocess to GCN1
    (0.4, 0.55, 0.5, 0.45),  # Preprocess to Skip1
    (0.6, 0.65, 0.7, 0.65),  # GCN1 to GCN2
    (0.6, 0.45, 0.7, 0.45),  # Skip1 to Skip2
    (0.8, 0.65, 0.9, 0.55),  # GCN2 to Postprocess
    (0.8, 0.45, 0.9, 0.55),  # Skip2 to Postprocess
    (1.0, 0.55, 1.1, 0.55),  # Postprocess to Output
]

for x1, y1, x2, y2 in connections:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

# Draw boxes for layers (on top of the arrows)
layers = [
    {"x": 0.1, "y": 0.5, "text": "Input\n(X, E, W)"},
    {"x": 0.3, "y": 0.5, "text": "Preprocess\n(create_ffn)"},
    {"x": 0.5, "y": 0.6, "text": "Graph Conv Layer 1"},
    {"x": 0.5, "y": 0.4, "text": "Skip Connection"},
    {"x": 0.7, "y": 0.6, "text": "Graph Conv Layer 2"},
    {"x": 0.7, "y": 0.4, "text": "Skip Connection"},
    {"x": 0.9, "y": 0.5, "text": "Postprocess\n(create_ffn)"},
    {"x": 1.1, "y": 0.5, "text": "Output\n(Logits)"},
]

for layer in layers:
    draw_box(ax, layer["x"], layer["y"], layer["text"])

# Set plot limits and remove axes
ax.set_xlim(0, 1.3)
ax.set_ylim(0.2, 0.8)
ax.axis('off')

# Add title
plt.title("Simplified GNN Architecture", fontsize=14, fontweight='bold')
plt.show()
