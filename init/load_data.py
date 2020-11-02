import sys
sys.path.append("../")
import os
import numpy as np
import pandas as pd
import pymatgen as mg
from sklearn.decomposition import PCA
from tsne import bh_sne
from matminer.featurizers.composition import ElementProperty
from matminer.utils.conversions import str_to_composition

class One_hot_matrix(object):
    
    def __init__(self, Periodic_table, point_len):
        self.Periodic_table = Periodic_table
        self.point_len = point_len
    
    def __call__(self, formulas):

        if isinstance(formulas, str):
            formulas = [formulas]
        reduce_forms = []
        for i, formula in enumerate(formulas):
            comp = mg.Composition(formula)
            reduce_form = comp.get_el_amt_dict()
            reduce_forms.append(reduce_form)

        one_hot_matrix = []
        for ind, formula in enumerate(reduce_forms):
            matrix = np.zeros(
                (1, len(self.Periodic_table), self.point_len)
             )
            keys = formula.keys()
            for symbols in keys:
                symbols_index = list(self.Periodic_table).index(symbols)
                matrix[0, symbols_index, int(formula[symbols]) - 1] = 1
            one_hot_matrix.append(matrix)
        one_hot_matrix = np.concatenate(one_hot_matrix, axis = 0)

        return one_hot_matrix

class One_hot_vec(object):

    def __init__(self, Periodic_table):
        self.Periodic_table = Periodic_table

    def __call__(self, formulas):
        if isinstance(formulas, str):
            formulas = [formulas]
        reduce_forms = []
        for i, formula in enumerate(formulas):
            comp = mg.Composition(formula)
            reduce_form = comp.get_el_amt_dict()
            reduce_forms.append(reduce_form)
        one_hot_vec = []
        for ind, formula in enumerate(reduce_forms):
            vec = np.zeros((1, len(self.Periodic_table)))
            keys = formula.keys()
            for symbols in keys:
                symbols_index = list(self.Periodic_table).index(symbols)
                vec[0, symbols_index] = float(formula[symbols])
            one_hot_vec.append(vec)
        one_hot_vec = np.concatenate(one_hot_vec, axis = 0)
        return one_hot_vec

def Magpie(formulas):
    if isinstance(formulas, str):
        formulas = [formulas]
    ep_feat = ElementProperty.from_preset(preset_name = "magpie")
    df = pd.DataFrame({"formula":formulas})
    df["composition"] = df["formula"].transform(str_to_composition)
    df = ep_feat.featurize_dataframe(df, col_id = "composition")
    df.drop(labels = ["composition", "formula"], axis = 1, inplace = True)
    return np.array(df).astype(np.float32)

class Atom2vec(object):
    def __init__(self, atom2vec_path):
        data = np.loadtxt(atom2vec_path, delimiter = ",", dtype = str)
        self.atom = data[:,0]
        self.atom_fec = data[:,1:].astype(np.float64)

    def __call__(self, formulas):
        if isinstance(formulas, str):
            formulas = [formulas]
        reduce_forms = []
        for i, formula in enumerate(formulas):
            comp = mg.Composition(formula)
            reduce_form = comp.get_el_amt_dict()
            reduce_forms.append(reduce_form)
        atom2vec = []
        for ind, formula in enumerate(reduce_forms):
            matrix = 0
            keys = formula.keys()
            for symbols in keys:
                symbols_index = list(self.atom).index(symbols)
                matrix += self.atom_fec[symbols_index]*float(formula[symbols])
            atom2vec.append(matrix[np.newaxis, :])
        atom2vecs = np.concatenate(atom2vec, axis = 0)        
        return atom2vecs

class Collate_batch(object):
    def __init__(self, idx, loop = True):
        
        self.idx = idx
        self.idx_epoch = 0
        self.epochs = 0
        self.end_one_epoch = False
        self.loop = loop
        self.loop_time = 0
        
    def next_batch(self, batch_size):
        start = self.idx_epoch
        if start == 0 or start + batch_size > len(self.idx):
            if start + batch_size > len(self.idx):
                self.epochs += 1
                if not self.loop:
                    self.end_one_epoch = True
            self.batch_idx = np.arange(len(self.idx))
            np.random.shuffle(self.batch_idx)
            self.idx_epoch = 0
            start = self.idx_epoch

        self.idx_epoch += batch_size
        end  = self.idx_epoch
        self.loop_time += 1
        return self.idx[self.batch_idx[start:end]]

    def __len__(self):
        return len(self.idx)

    def __call__(self):
        return self.idx

class Dataset(Collate_batch):
    def __init__(self, data_path, fea_type, Periodic_table, formula = "formulas", target = None):
        self.data_path = data_path
        self.fea_type = fea_type
        self.Periodic_table = list(np.loadtxt(Periodic_table, dtype = str))
        self.data = np.array(
            pd.read_csv(self.data_path, keep_default_na = False)[formula]
        )
        self.target = target
        if target != None:
            self.y = np.array(
                pd.read_csv(self.data_path, keep_default_na = False)[target]
            )
        else:
            self.y = None

        if self.fea_type == "magpie":
            self.magpie_fea = Magpie(self.data)

        super(Dataset, self).__init__(np.arange(len(self.data)), False)

    def __getitem__(self, idx):
        if self.fea_type == "one_hot_matrix":
            fea = One_hot_matrix(self.Periodic_table, point_len = 8)
        if self.fea_type == "magpie":
            fea = self.magpie_fea
        if self.fea_type == "atom2vec":
            fea = Atom2vec("./atom2vec/" + "OQMD" + "/atom2vec.csv")

        if self.fea_type == "one_hot_vec":
            fea = One_hot_vec(self.Periodic_table)
        
        if self.target == None:
            if self.fea_type == "magpie":
                return self.data[idx], fea[idx]
            else:
                return self.data[idx], fea(self.data[idx])
        else:
            if self.fea_type == "magpie":
                return self.data[idx], fea[idx], self.y[idx]
            else:
                return self.data[idx], fea(self.data[idx]), self.y[idx]
    def __len__(self):
        return len(self.data)
    
def Subset(data, per, seed = None):
    idx = np.arange(len(data))
    if seed == None:
        np.random.seed()
        np.random.shuffle(idx)
    else:
        np.random.seed(seed)
        np.random.shuffle(idx)
    train_idx = idx[:int(len(data)*per)]
    test_idx = idx[int(len(data)*per):]
    train = Collate_batch(train_idx, False)
    test = Collate_batch(test_idx, False)
    return train, test

def One_hot_fea(Periodic_table):
    Periodic_table = list(np.loadtxt(Periodic_table, dtype = str))
    one_hot_fea = np.zeros((len(Periodic_table), len(Periodic_table)))
    for i, atom in enumerate(Periodic_table):
        atom_idx = Periodic_table.index(atom)
        one_hot_fea[atom_idx][atom_idx] = 1
    return one_hot_fea

class Evalution(object):
    def __init__(self):
        self.total_loss = 0
    def updata(self, loss):
        self.total_loss += np.abs(loss)
    def lotal_loss(self):
        return self.total_loss

class Dim_reduce(object):

    def __init__(self, way = "tsne"):
        self.way = way
        if self.way == "tsne":
            self.red_dim = bh_sne
        if self.way == "pca":
            self.red_dim = PCA(n_components = 2).fit_transform
    def __call__(self, feature):
        if self.way == "pca":
            return self.red_dim(feature)
        else:
            return self.red_dim(perplexity = 3, data = feature)

def reduce_molecule(molecular_map, periodic_table, tolerate):
    periodic_table = np.loadtxt(periodic_table, dtype = str)
    all_for_str, all_for_dic = [], []
    for index, molecular in enumerate(molecular_map):
        formula_str, formula_dic = "", dict()
        for ind, one_vec in enumerate(molecular):
            for j, num in enumerate(one_vec):
                if (num >= (1 - tolerate)) and (num <= 1):
                    formula_str += periodic_table[ind] + str(j + 1)
                    formula_dic[periodic_table[ind]] = j + 1
        all_for_str.append(formula_str)
        all_for_dic.append(formula_dic)
    return np.array(all_for_str), np.array(all_for_dic)
