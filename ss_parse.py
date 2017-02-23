import pickle

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

def prepare_data():
    with open("ss.txt") as handle:
    records = list(ss_parser(handle))
    sequences = [seq.replace("U", "X").replace("B", "X").replace("Z", "X").replace("O", "X") for seq in records[0::2]]
    structures = records[1::2]

    with open('sequences.pickle', 'wb') as handle:
        pickle.dump(sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('structures.pickle', 'wb') as handle:
        pickle.dump(structures, handle, protocol=pickle.HIGHEST_PROTOCOL)