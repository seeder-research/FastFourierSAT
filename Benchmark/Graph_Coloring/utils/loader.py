class Formula(object):
    def __init__(self):
        self._xor_claus  = []
        self._cnf_claus  = []
        self._eo_claus = []

    def read_DIMACS(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                split = line.split()
                if len(line) == 0 or split[0] == 'c' or split[0] == '*':
                    continue
                if (split[0] == 'p'):    # Read p line
                    self._n_var = int(line.split()[2])
                else:
                    clause_type = split[1]
                    lits = [int(val) for val in split[2:-1]]
                    if split[1] == 'eo':   # Exactly-1
                        self._eo_claus.append(lits)
                    elif split[1] == 'x':    # XOR
                        self._xor_claus.append(lits)
                    elif split[1] == 'cnf':
                        self._cnf_claus.append(lits)