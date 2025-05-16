import pandas as pd
import matplotlib.pyplot as plt
import os

#load
graph = pd.read_csv('VAE_loss.csv')

#making sure it loads
print(graph.head())

#create it
plt.figure(figsize=(14, 7))
plt.plot(graph['step'], graph['nelbo'], label='NELBO') # could add linewidth=0.7 to make smaller lines but unnec
plt.plot(graph['step'], graph['rec'], label='REC')
plt.plot(graph['step'], graph['kl'], label='KL')

#labels & title
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Losses Over Steps')
plt.legend()
plt.grid(True)

#@show
plt.show()
plt.savefig('loss_plot.png', dpi=300)
