class Formula(object):
    def __init__(self):
        self._xor_claus  = []
        self._card_claus = []
        self._card_k     = []

    def read_DIMACS(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                split = line.split()
                if len(line) == 0 or split[0] == 'c' or split[0] == '*':
                    continue
                if (split[0] == 'p'):    # Read p line
                    self._n_var = int(line.split()[2])
                elif line[0] == 'x':
                    lits = map(int, [split[i] for i in range(1, len(split) - 1)])
                    lits = list(lits)
                    self._xor_claus.append(lits)
                elif line[0] == 'd':
                    self._card_k.append(-int(split[1]))
                    lits = map(int, [split[i] for i in range(2, len(split) - 1)])
                    lits = list(lits)
                    self._card_claus.append(lits)