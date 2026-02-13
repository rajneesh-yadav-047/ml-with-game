import matplotlib.pyplot as plt
import numpy as np
import torch

# Initialize figures and axes for both plots
main_fig, axs = None, None

def init_plot():
    global main_fig, axs
    if main_fig is None:
        plt.ion()
        main_fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        main_fig.canvas.manager.set_window_title('Evolution Dashboard')
        main_fig.tight_layout(pad=3.0)

def plot(scores, mean_scores):
    init_plot()
    score_ax = axs[0, 0]

    score_ax.cla() # Clear the score axis
    score_ax.set_title('Training Progress')
    score_ax.set_xlabel('Number of Games / Generations')
    score_ax.set_ylabel('Score')
    score_ax.plot(scores, label='Score')
    score_ax.plot(mean_scores, label='Mean Score')
    score_ax.set_ylim(ymin=0)
    score_ax.legend()
    if scores:
        score_ax.text(len(scores)-1, scores[-1], str(scores[-1]))
    if mean_scores:
        score_ax.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 2)))
    
    main_fig.canvas.draw()
    main_fig.canvas.flush_events()

def plot_weights(model):
    init_plot()
    weight_ax = axs[0, 1]

    weight_ax.cla() # Clear the weight axis
    weight_ax.set_title('Weight Distribution of Best Model')
    weight_ax.set_xlabel('Weight Value')
    weight_ax.set_ylabel('Frequency')
    
    weights = []
    for param in model.parameters():
        # .detach() is good practice when not training
        weights.extend(param.cpu().detach().numpy().flatten())
        
    weight_ax.hist(weights, bins=50, color='orange', alpha=0.7)
    
    main_fig.canvas.draw()
    main_fig.canvas.flush_events()

def plot_network_graph(model):
    init_plot()
    network_ax = axs[1, 0] # Spanning logic is complex in simple subplots, let's use bottom left
    # To make it span, we would need gridspec, but let's keep it simple for now or use the 3rd slot.

    network_ax.cla()
    network_ax.set_title('Network Graph of Best Model')
    network_ax.axis('off')

    # Get layer sizes from the model
    layer_sizes = []
    for layer in model.net:
        if isinstance(layer, torch.nn.Linear):
            if not layer_sizes:
                layer_sizes.append(layer.in_features)
            layer_sizes.append(layer.out_features)
    
    # Calculate node positions
    v_spacing = 1.0
    h_spacing = 2.0
    node_positions = []
    for i, size in enumerate(layer_sizes):
        y_positions = np.linspace(-v_spacing * (size - 1) / 2., v_spacing * (size - 1) / 2., size)
        node_positions.append([(i * h_spacing, y) for y in y_positions])

    # Draw connections (lines)
    for i in range(len(layer_sizes) - 1):
        weights = list(model.parameters())[i*2].cpu().detach().numpy()
        for j, (x_start, y_start) in enumerate(node_positions[i]):
            for k, (x_end, y_end) in enumerate(node_positions[i+1]):
                weight = weights[k, j]
                color = 'red' if weight < 0 else 'blue'
                alpha = min(1.0, abs(weight) * 0.5) # Scale alpha to reduce clutter
                network_ax.plot([x_start, x_end], [y_start, y_end], color=color, alpha=alpha, lw=0.5)

    # Draw nodes (circles)
    for i, layer in enumerate(node_positions):
        x_coords = [pos[0] for pos in layer]
        y_coords = [pos[1] for pos in layer]
        color = 'lightblue' # Hidden layer
        if i == 0: color = 'lightgreen' # Input layer
        if i == len(layer_sizes) - 1: color = 'lightcoral' # Output layer
        network_ax.scatter(x_coords, y_coords, s=150, c=color, ec='black', zorder=5)

    network_ax.relim()
    network_ax.autoscale_view()
    main_fig.canvas.draw()
    main_fig.canvas.flush_events()