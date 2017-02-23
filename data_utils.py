import pickle
import numpy as np

# dictionary for amino acids
res_to_num = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9,  
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20}

# dictionary for structures
struct_to_num = {' ': 0, 'H': 1, 'B': 2, 'E': 3, 'G': 4, 'I': 5, 'T': 6, 'S': 7} 

# returns a one-hot representation of a sequence
def one_hot(seq, to_num, max_len):
    rep_size = len(to_num)
    def o_h(aa):
        out = np.zeros(rep_size)
        out[to_num[aa]] = 1
        return out
    oh = np.array([o_h(aa) for aa in seq])
    padding = np.zeros((max_len - len(seq), rep_size))
    return np.concatenate((padding, oh), axis=0)

# returns a sequence of one-hot representations
def one_hot_seq(sequences, to_num, max_len):
    return np.array([one_hot(seq, to_num, max_len) for seq in sequences])

# returns generator for protein and secondary structure data, to help with memory usage
def generator(sequences, structures, max_len, batch_size=128):
    starting_index = 0
    while True:
        while starting_index < len(sequences):
            batch_sequences = sequences[starting_index: starting_index + batch_size]
            batch_structures = structures[starting_index: starting_index + batch_size]
            starting_index += batch_size
            yield one_hot_seq(batch_sequences, res_to_num, max_len), one_hot_seq(batch_structures, struct_to_num, max_len)
        starting_index = 0

# parses ss.txt
def ss_parser(handle):
    while True:
        line = handle.readline()
        if line == "":
            return
        if line[0] == ">":
            break
    while True:
        lines = []
        line = handle.readline()
        while True:
            if not line:
                break
            if line[0] == ">":
                break
            lines.append(line)
            line = handle.readline()
        yield "".join(lines).replace("\r", "").replace("\n", "")
        if not line:
            return

# saves protein and structure data in separate pickle files
def prepare_data():
    with open("ss.txt") as handle:
        records = list(ss_parser(handle))
        sequences = [seq.replace("U", "X").replace("B", "X").replace("Z", "X").replace("O", "X") for seq in records[0::2]]
        structures = records[1::2]

    with open('sequences.pickle', 'wb') as handle:
        pickle.dump(sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('structures.pickle', 'wb') as handle:
        pickle.dump(structures, handle, protocol=pickle.HIGHEST_PROTOCOL)