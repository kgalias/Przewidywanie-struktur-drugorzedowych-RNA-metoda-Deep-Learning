import numpy as np
res_to_num = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9,  
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20}

struct_to_num = {' ': 0, 'H': 1, 'B': 2, 'E': 3, 'G': 4, 'I': 5, 'T': 6, 'S': 7} 

def one_hot(seq, to_num, max_len):
    rep_size = len(to_num)
    def o_h(aa):
        out = np.zeros(rep_size)
        out[to_num[aa]] = 1
        return out
    oh = np.array([o_h(aa) for aa in seq])
    padding = np.zeros((max_len - len(seq), rep_size))
    return np.concatenate((padding, oh), axis=0)

def one_hot_seq(sequences, to_num, max_len):
    return np.array([one_hot(seq, to_num, max_len) for seq in sequences])

def generator(sequences, structures, max_len, batch_size=128):
    starting_index = 0
    #max_len = max([len(seq) for seq in sequences])
    while True:
        while starting_index < len(sequences):
            batch_proteins = sequences[starting_index: starting_index + batch_size]
            batch_structures = structures[starting_index: starting_index + batch_size]
            starting_index += batch_size
            yield one_hot_seq(batch_proteins, res_to_num, max_len), one_hot_seq(batch_structures, struct_to_num, max_len)
        starting_index = 0