import seaborn as sns
import matplotlib.pyplot as plt

data = [
    [3, 1, 44, 57],
    [0, 0, 100, 100],
    [4, 0, 100, 0],
    [0, 0, 100, 100]
]
custom_x_labels = ['Heuristic', 'Random', 'Offline', 'Self-Play']
b = sns.color_palette("vlag", as_cmap=True)
ax = sns.heatmap(data, annot=True, cmap=b, cbar_kws={'label': 'Percentage'}, fmt=".0f", annot_kws={"size": 14},
                 xticklabels=custom_x_labels, yticklabels=custom_x_labels)
ax.set(xlabel="", ylabel="")
# ax.xaxis.tick_top()
cbar_axes = ax.figure.axes[-1]
cbar_axes.yaxis.label.set_size(12)
plt.plot([1, 1], [0, 1], color='k', linestyle='-', linewidth=5)
plt.plot([1, 2], [1, 1], color='k', linestyle='-', linewidth=5)
plt.plot([2, 2], [1, 2], color='k', linestyle='-', linewidth=5)
plt.plot([2, 3], [2, 2], color='k', linestyle='-', linewidth=5)
plt.plot([3, 3], [2, 3], color='k', linestyle='-', linewidth=5)
plt.plot([3, 4], [3, 3], color='k', linestyle='-', linewidth=5)
plt.xlabel('Model A Checkpoint', fontsize=14)
plt.ylabel('Model B Checkpoint', fontsize=14)
plt.title('Win and Draw Rates During Training', fontsize=16)
plt.savefig('heatplot.png', dpi=300, bbox_inches='tight')
