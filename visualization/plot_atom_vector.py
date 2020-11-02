import numpy as np
import sys
sys.path.append("../")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.loadtxt("../atom2vec/" + "OQMD" + "/atom2vec.csv", delimiter = ",", dtype = str)
atom = data[:,0]
atom_fec = data[:,1:].astype(np.float64)

plt.figure(figsize = (80, 20))
ax = plt.gca()
im = ax.imshow(atom_fec[:,:20], cmap = "BrBG_r")
cbar = ax.figure.colorbar(im, ax = ax, shrink = 0.3, aspect = 20, pad = 0.005)

ax.set_xticks(np.arange(20))
ax.set_yticks(np.arange(atom.shape[0]))
ax.set_yticklabels(atom)

for edge, spine in ax.spines.items():
	spine.set_visible(False)
ax.set_xticks(np.arange(20+1)-.5, minor=True)
ax.set_yticks(np.arange(atom.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth = 0.3)
ax.tick_params(which = "minor", bottom = False, left = False)
plt.savefig("1.png",bbox_inches = "tight")

