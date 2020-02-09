import numpy as np

def indent_string(str, indent_spaces):
    return " "*indent_spaces + str.replace("\n", ("\n")+" "*indent_spaces)

def keywithmaxval(d):
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(max(v))]

def str_to_bool(s):
    s = s.replace(" ","").lower()
    return s in ["y", "yes", "true", "1"]

def one_hot(idx, n_dim=8):
    x = np.zeros(n_dim, dtype=int)
    x[idx] = 1
    return x
