import sys
sys.path.append("../")
import numpy as np
from init.load_data import Dataset, Dim_reduce
from init.config import parse_args
import matplotlib.pyplot as plt

def scatter_plot(data, label, name):
    a, b, c, d = 0, 0, 0, 0
    for i, xy in enumerate(data):
        if ("Cu" in label[i]) and ("Fe" not in label[i]):
            if a == 1:
                plt.scatter(xy[0], xy[1], color = "#3A5FCD", marker = "s", s = 5, label = "Cuprates")
                a +=1 
            else:
                plt.scatter(xy[0], xy[1], color = "#3A5FCD", marker = "s", s = 5)
                a += 1
        if ("Fe" in label[i]) and ("Cu" not in label[i]):
            if b == 1:
                plt.scatter(xy[0], xy[1], color = "#EE0000", marker = "v", s = 2, label = "Iron-based")
                b += 1
            else:
                plt.scatter(xy[0], xy[1], color = "#EE0000", marker = "v", s = 2)
                b += 1
        if ("Cu" not in label[i]) and ("Fe" not in label[i]):
            if c == 1:
                plt.scatter(xy[0], xy[1], color = "black", marker = "x", s = 2, label = "Other")
                c += 1
            else:
                plt.scatter(xy[0], xy[1], color = "black", marker = "x", s = 2)
                c += 1
        # if ("Fe" in label[i]) and ("Cu" in label[i]):
        #     if d == 1:
        #         plt.scatter(xy[0], xy[1], color = "#00FF00", marker = "o", s = 2, label = "Cuprates and Iron-based")
        #         d += 1
        #     else:
        #         plt.scatter(xy[0], xy[1], color = "#00FF00", marker = "o", s = 2)
        #         d += 1
    plt.legend(shadow = True, numpoints = 1, framealpha = 1, frameon = True, edgecolor = "black")
    plt.savefig(name + ".pdf", bbox_inches = "tight")
    plt.show()

def main(args):
    data = Dataset("../data/superconder/supercon-tc.csv", "atom2vec", "../data/Periodic_Table/Periodic_Table.csv", "formulas", "bandgap")[:]
    formulas, fea = data[0], data[1].astype(np.float64)
    clust = Dim_reduce("pca")
    data_2d = clust(fea)
    scatter_plot(data_2d, formulas, "pca_atom2vec")
if __name__ == "__main__":
    main(parse_args())