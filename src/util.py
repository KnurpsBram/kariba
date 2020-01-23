import numpy as np

animal_names = [
    ("mouse", "mice"),
    ("meerkat", "meerkats"),
    ("zebra", "zebras"),
    ("giraffe", "giraffes"),
    ("ostrich", "ostriches"),
    ("leopard", "leopards"),
    ("rhino", "rhinos"),
    ("elephant", "elephants")
]
animal_str_to_idx = {animal_names[i][j] : i for i in range(8) for j in range(2)} # call animal_str_to_idx["zebra"] for 2
animal_idx_to_str = {i : {"singular" : animal_names[i][0], "plural" : animal_names[i][1]} for i in range(8)} # call animal_idx_to_str[6]["plural"] for "leopards"

def indent_string(str, indent_spaces):
    return " "*indent_spaces + str.replace("\n", ("\n")+" "*indent_spaces)

def animals_arr_to_str(arr):
    # translate [0, 0, 1, 0, 0, 2, 0, 0] to "2 leopards 1 zebra"
    if np.sum(arr) == 0:
        return "nothing\n"
    else:
        return "\n".join([str(arr[i]) + " " + (animal_idx_to_str[i]["singular"] if arr[i] == 1 else animal_idx_to_str[i]["plural"]) for i in reversed(range(8)) if arr[i] > 0])

def action_str_to_arr(s):
    try:
        s = s.replace(" ","").lower() # remove whitespace and case-invariant
        if s.isdigit() and len(s) == 8: # if typed like 00030000 for '3 giraffes'
            action = np.array([int(c) for c in s], dtype=int) # change to array
        if "*" in s: # if typed like 3*4 for '3 giraffes'
            n = int(s[0])
            animal_idx = int(s[2])
            action = n*one_hot(animal_idx)
        else:
            n = int(s[0])
            animal_idx = animal_str_to_idx[s[1:]]
            action = n*one_hot(animal_idx)
    except:
        action = np.zeros(8)
    return action

def str_to_bool(s):
    s = s.replace(" ","").lower()
    return s in ["y", "yes", "true", "1"]

def one_hot(idx, n_dim=8):
    x = np.zeros(n_dim, dtype=int)
    x[idx] = 1
    return x

class NodenameGenerator():
    def __init__(self):
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        return "Node"+str(self.i)
