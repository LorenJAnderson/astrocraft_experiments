import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

checkpoints = [0, 1, 2, 3, 4, 5, 6, 7, 8]
win_rate = [728, 1259, 1248, 1237, 1254, 1230, 1246, 1241, 1263]
ax1 = sns.lineplot(x=checkpoints, y=win_rate, c='#2369BD')
plt.plot([-0.3, 8.3], [1245, 1245], color='k', linestyle=':', linewidth=1)
plt.text(0.3, 1230, 'heuristic', horizontalalignment='center', verticalalignment='center', fontweight='book', fontsize=10)
plt.xlabel('Checkpoint', fontsize=14)
plt.ylabel('Elo', fontsize=14)
plt.title('Elo Rating Throughout Training', fontsize=16)
plt.savefig('elo.png', dpi=300, bbox_inches='tight')
