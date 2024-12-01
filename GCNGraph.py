# Further spaced-out GCN Architecture Visualization for enhanced readability

import matplotlib.pyplot as plt

# Create the figure and axis
fig, ax = plt.subplots(figsize=(16, 12))

# Function to draw rectangles representing layers
def draw_layer(ax, x, y, text, width=0.4, height=0.15, color="lightblue"):
    ax.add_patch(plt.Rectangle((x, y), width, height, edgecolor="black", facecolor=color, lw=1.5))
    ax.text(x + width / 2, y + height / 2, text, fontsize=9, ha="center", va="center", wrap=True)

# Function to draw arrows connecting layers
def draw_arrow(ax, start, end, style="->"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color="gray", lw=1.5))

# Define layers with even more spacing
layers = [
    {"x": 0.1, "y": 0.7, "text": "Input Layer\n(Node Features, \nAdjacency Matrix)"},
    {"x": 0.6, "y": 0.7, "text": "Preprocess Layer\n(Dense, ReLU, \nL2 Regularization)"},
    {"x": 1.1, "y": 0.85, "text": "Graph Conv Layer 1\n(Aggregation, \nLeaky ReLU, Dropout)"},
    {"x": 1.1, "y": 0.55, "text": "Graph Conv Layer 2\n(Aggregation, \nLeaky ReLU, Dropout)"},
    {"x": 1.6, "y": 0.7, "text": "Hidden Layers\n(Dense, ReLU)"},
    {"x": 2.1, "y": 0.7, "text": "Postprocess Layer\n(Dense, ReLU, Dropout)"},
    {"x": 2.6, "y": 0.7, "text": "Output Layer\n(Logits, Softmax)"}
]

# Draw all layers
for layer in layers:
    draw_layer(ax, layer["x"], layer["y"], layer["text"])

# Define arrows for connections between layers
arrows = [
    ((0.5, 0.75), (0.6, 0.75)),  # Input to Preprocess
    ((1.0, 0.75), (1.1, 0.9)),  # Preprocess to GCN Layer 1
    ((1.0, 0.75), (1.1, 0.6)),  # Preprocess to GCN Layer 2
    ((1.5, 0.9), (1.6, 0.75)),  # GCN Layer 1 to Hidden Layers
    ((1.5, 0.6), (1.6, 0.75)),  # GCN Layer 2 to Hidden Layers
    ((2.0, 0.75), (2.1, 0.75)),  # Hidden Layers to Postprocess
    ((2.5, 0.75), (2.6, 0.75))   # Postprocess to Output
]

# Draw skip connections
skip_connections = [
    ((0.5, 0.75), (1.1, 0.9)),  # Skip connection: Preprocess to GCN Layer 1
    ((0.5, 0.75), (1.1, 0.6))   # Skip connection: Preprocess to GCN Layer 2
]

# Draw main arrows
for start, end in arrows:
    draw_arrow(ax, start, end)

# Draw skip arrows (dashed for clarity)
for start, end in skip_connections:
    draw_arrow(ax, start, end, style="-|>")

# Set plot limits and remove axes
ax.set_xlim(0, 3.2)
ax.set_ylim(0.4, 1.0)
ax.axis('off')

# Add title
plt.title("Graph Convolutional Network (GCN) Architecture", fontsize=14, fontweight='bold')
plt.show()
