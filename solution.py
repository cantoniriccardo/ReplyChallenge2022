import os

import numpy as np


class Solution:

    def __init__(self, problem):
        #problem.compute_total_reward()
        self.problem = problem
        #self.sol = np.array([e for e in range(problem.D) if problem.Ntot_d[e] > 0])  # order of killed enemies
        self.sol = np.array([e for e in range(problem.D)])  # order of killed enemies
        self.stamina = np.zeros(problem.T + 1, dtype=np.int32)  # Stamina at the beginning of each turn, the last value is the residual stamina
        self.n_killed = 0  # number of enemies killed
        self.reward = 0

    # compute T, R, n_killed from E
    def score(self):
        problem = self.problem
        self.stamina[:] = 0
        self.stamina[0] = problem.Si
        self.n_killed = 0
        reward = 0

        # Update T, R and n_killed
        enemy_index = 0
        for t in range(0, problem.T):
            next_enemy = self.sol[enemy_index]
            if self.stamina[t] > problem.Smax:
                self.stamina[t] = problem.Smax
            self.stamina[t + 1] += self.stamina[t]
            if self.stamina[t] >= problem.Sc_d[next_enemy]:
                enemy_index += 1

                # Update the stamina
                self.stamina[t + 1] -= problem.Sc_d[next_enemy]
                recover_turn = t + problem.Tr_d[next_enemy]
                if recover_turn < problem.T:
                    self.stamina[recover_turn] += problem.Sr_d[next_enemy]

                # Compute the reward
                rewarding_turns = problem.Nf_d[next_enemy]
                if rewarding_turns > 0:
                    remaining_turns = problem.T - t
                    last_reward_i = remaining_turns if remaining_turns < rewarding_turns else rewarding_turns
                    reward += problem.Ncum_d[next_enemy][last_reward_i - 1]

        self.n_killed = enemy_index
        self.reward = reward

        return reward

    def remaining_enemies(self):
        return self.sol[self.n_killed:]

    def dump(self):
        # save the file in the solution directory. It not exist create it
        if not os.path.exists("solution"):
            os.makedirs("solution")
        score = self.reward
        filename = f"solution/{self.problem.name}-{score}.txt"
        with open(filename, "w") as f:
            lines = [str(e) for e in self.sol]
            lines += [str(e) for e in set(range(self.problem.D)) - set(self.sol)]
            f.write("\n".join(lines))

    def load(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()
            order = np.array([int(e) for e in lines[:self.problem.D]])
            self.sol = np.array([e for e in order] + [e for e in set(range(self.problem.D)) - set(order)])
