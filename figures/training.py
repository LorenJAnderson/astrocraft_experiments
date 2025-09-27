import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

checkpoints = [0, 1, 2, 3, 4, 5, 6, 7, 8]
win_rate = [1, 53, 51, 48, 46, 53, 47, 47, 62]
draw_rate = [0, 3, 3, 3, 2, 2, 0, 3, 2]

ax1 = sns.lineplot(x=checkpoints, y=win_rate, c='#AC3F42', label='Win Rate')
ax2 = sns.lineplot(x=checkpoints, y=draw_rate, c='#2369BD', label='Draw Rate')
ax2.lines[1].set_linestyle("--")
plt.xlabel('Checkpoint', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.title('Evaluation Against Heuristic Opponent', fontsize=16)
plt.legend()
plt.plot([4, 4], [0, 65], color='k', linestyle=':', linewidth=1)
plt.text(5.5, 20, r'online training $\longrightarrow$', horizontalalignment='center', verticalalignment='center', fontweight='book', fontsize=10)
plt.text(2.5, 20, r'$\longleftarrow$ offline training', horizontalalignment='center', verticalalignment='center', fontweight='book', fontsize=10)
plt.savefig('training.png', dpi=300, bbox_inches='tight')

