import sys
sys.path.append("../")
from init.config import parse_args
from init.load_data import Dim_reduce
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot(args, atoms, data_2d):
    # plt.scatter(train_data[:,0][0], train_data[:,1][0], color = "#3A5FCD", marker = "s", s = 1, label = "Training")
    # plt.savefig("args.png", bbox_inches = "tight", dpi = 1500)
    
    for i, atom in enumerate(atoms):
        plt.scatter(data_2d[i][0], data_2d[i][1], color = "#3A5FCD", marker = "s", s = 1)
        plt.text(data_2d[i][0] + 0.02, data_2d[i][1] + 0.02, atom)
    plt.savefig(args.database + ".png")

def main(args):
    atom2vec_path = "../atom2vec/" + args.database + "/atom2vec.csv"
    data = np.loadtxt(atom2vec_path, delimiter = ",", dtype = str)
    atom = data[:,0]
    atom_fec = data[:,1:].astype(np.float64)
    # tsne = TSNE(n_components=2)
    # tsne.fit_transform(atom_fec)
    # data_2d = tsne.embedding_
    clust = Dim_reduce("tsne")
    data_2d = clust(atom_fec)
    plot(args, atom, data_2d)

if __name__ == "__main__":
    main(parse_args())

