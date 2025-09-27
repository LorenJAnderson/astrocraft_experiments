import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# Set data
df = pd.DataFrame({
    'group': ['A', 'B', 'C'],
    'Draw Rate': [0.0, 0.33, 1],
    '          Aggression': [0, 1, 0.96],
    '     Fuel Use': [1, 0.466, 0.441],
    'Max Orbital': [1, 0.0, 0.0],
    'Action Entropy               ': [1, 0.110, 0.110],
    'Episode Length                ': [0.681, 1, 0.978]
})

# ------- PART 1: Create background

# number of variable
categories = list(df)[1:]
N = len(categories)

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels
print(angles[:-1])
plt.xticks(angles[:-1], categories)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=9)
plt.ylim(0, 1)

# ------- PART 2: Add plots

# Plot each individual = each line of the data

# Ind1
values = df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Random", color='#2369BD')
ax.lines[0].set_linestyle("--")
ax.fill(angles, values, alpha=0.1, color='#2369BD')

# Ind2
values = df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Offline", color='#c944ff')
ax.lines[1].set_linestyle(":")
ax.fill(angles, values, alpha=0.1, color='#c944ff')

# Ind3
values = df.loc[2].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Self-Play", color='#AC3F42')
ax.fill(angles, values, alpha=0.1, color='#AC3F42')

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), title='Checkpoint')
plt.title('Quantified Behaviors Throughout Training', fontsize=16)

# Show the graph
# plt.show()
plt.savefig('radar.png', dpi=300, bbox_inches='tight')
