# ---------------------- Matplotlib Architecture Theory ----------------------
# Matplotlib is a powerful Python library for data visualization.
# It has a hierarchy of objects: 
#   1. Figure → The entire window or page where everything is drawn.
#   2. Axes → A single plot inside the figure (can have x/y axis, title, etc.)
#   3. Subplot → A type of Axes arranged in a grid layout.
#   4. Artists → Everything you see on the plot (lines, text, legends, etc.)

# Main modules:
# - matplotlib.pyplot: High-level interface, similar to MATLAB-style plotting.
# - matplotlib.figure: Handles the overall figure.
# - matplotlib.axes: Manages the individual subplots (plots with axis).
# - matplotlib.gridspec: Lets you create flexible grid-based layouts for subplots.

# Important Terminologies:
# - Figure: The whole canvas/window that holds one or more subplots.
# - Axes: An individual plot (not the X/Y lines) – it's where data is plotted.
# - Subplot: A special type of axes arranged in a grid using methods like `add_subplot`.
# - GridSpec: A tool to customize the layout of subplots (rows, cols, and size ratios).

# ---------------------- Now the Code ----------------------

import matplotlib.pyplot as plt  # High-level API to create plots

# Create a figure object – this is the blank canvas where we add subplots.
fig = plt.figure(figsize=(10, 8))

# Use GridSpec to define a 2x2 layout:
# width_ratios: relative width of columns → both same here (5,5)
# height_ratios: first row (0.3) is smaller, second row (5) is larger
gs = fig.add_gridspec(2, 2, width_ratios=[5, 5], height_ratios=[0.3, 5])

# Now add subplots in specific grid cells using the GridSpec layout
ax1 = fig.add_subplot(gs[0, 0])  # Top-left cell
ax2 = fig.add_subplot(gs[0, 1])  # Top-right cell
ax3 = fig.add_subplot(gs[1, 0])  # Bottom-left cell (larger row)
ax4 = fig.add_subplot(gs[1, 1])  # Bottom-right cell (larger row)

# Add titles to each subplot to understand their placement
ax1.set_title('Top-Left')
ax2.set_title('Top-Right')
ax3.set_title('Bottom-Left (Larger)')
ax4.set_title('Bottom-Right (Larger)')

# Display everything
plt.show()
