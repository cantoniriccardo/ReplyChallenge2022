import numpy as np


class Problem:

    def __init__(self):
        self.name = "noname"
        self.T = 0  # turns
        self.Si = 0  # initial stamina
        self.Smax = 0  # max stamina
        self.D = 0  # number of enemies
        self.Sc_d = np.array([], dtype=np.int32)  # stamina cost of each enemy
        self.Tr_d = np.array([], dtype=np.int32)  # time to recover stamina
        self.Sr_d = np.array([], dtype=np.int32)  # stamina recovered
        self.Nf_d = np.array([], dtype=np.int32)  # number of rewards for each enemy
        self.Na_d = []  # array of reward for each enemy
        self.Ncum_d = []  # cumulative reward for each enemy
        self.Ntot_d = None  # total reward

        # Reward matrix
        self.A = []  # matrix of reward array, row enemy, column turn

    def compute_total_reward(self):
        self.Ntot_d = np.array([np.sum(self.Na_d[i]) for i in range(0, self.D)])

    def get_top_rewarding_enemies(self, n):
        if self.Ntot_d is None:
            self.compute_total_reward()
        return np.flip(np.argsort(self.Ntot_d)[-n:])

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        p = cls()
        p.name = filename.split("/")[-1].split(".")[0]
        p.Si, p.Smax, p.T, p.D = map(int, lines[0].split())
        p.Sc_d = np.zeros(p.D, dtype=np.int32)
        p.Tr_d = np.zeros(p.D, dtype=np.int32)
        p.Sr_d = np.zeros(p.D, dtype=np.int32)
        p.Nf_d = np.zeros(p.D, dtype=np.int32)
        p.Na_d = []
        for i, line in enumerate(lines[1:]):
            Sc_d, Tr_d, Sr_d, Na_f, *Na_d = map(int, line.split())
            p.Sc_d[i] = Sc_d
            p.Tr_d[i] = Tr_d
            p.Sr_d[i] = Sr_d
            p.Nf_d[i] = Na_f
            p.Na_d.append(np.array(Na_d, dtype=np.int32))
            p.Ncum_d.append(np.cumsum(p.Na_d[i]))
        return p
