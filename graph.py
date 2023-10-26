import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

df = pd.read_csv('dists.csv')
support = df['Unnamed: 0']

fig, ax = plt.subplots()
ps_a_data = df['ps_a']
m_data = df['m']
ax.bar(support, ps_a_data, align='edge', width=0.7)
ax.bar(support, m_data, align='edge', width=-0.7)

width = 0.5
def update(frame):
    plt.cla()
    try:
        df = pd.read_csv('dists.csv')
        ps_a_data = df['ps_a']
        m_data = df['m']
    except:
        ps_a_data = [0] * len(support)
        m_data = [0] * len(support)
    ax.bar(support - width / 2, m_data, align='center', color='blue')
    ax.bar(support + width / 2, ps_a_data, align='center', color='red')

ani = FuncAnimation(fig, update, interval=500)

plt.show()
