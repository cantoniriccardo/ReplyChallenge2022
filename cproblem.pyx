import numpy as np
import os
import time


cdef class CProblem:
    cdef public int[:] sol
    cdef public int[:] stamina
    cdef public int n_killed
    cdef public int reward

    cdef public str name
    cdef public int Smax
    cdef public int D
    cdef public int T
    cdef public int Si
    cdef public int[:] Sc_d
    cdef public int[:] Tr_d
    cdef public int[:] Sr_d
    cdef public int[:] Nf_d
    cdef public list Na_d
    cdef public list Ncum_d

    def init_sol(self):
        self.sol = np.array([e for e in range(self.D)], dtype=np.int32)  # order of killed enemies
        self.stamina = np.zeros(self.T + 1,
                                dtype=np.int32)  # Stamina at the beginning of each turn, the last value is the residual stamina
        self.n_killed = 0  # number of enemies killed
        self.reward = 0
        self.Ncum_d = self.Ncum_d

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
            p.Ncum_d = []
            for i, line in enumerate(lines[1:]):
                Sc_d, Tr_d, Sr_d, Na_f, *Na_d = map(int, line.split())
                p.Sc_d[i] = Sc_d
                p.Tr_d[i] = Tr_d
                p.Sr_d[i] = Sr_d
                p.Nf_d[i] = Na_f
                p.Na_d.append(np.array(Na_d, dtype=np.int32))
                p.Ncum_d.append(np.cumsum(p.Na_d[i]))
            p.init_sol()
            return p

    def dump(self):
        # save the file in the solution directory. It not exist create it
        if not os.path.exists("solution"):
            os.makedirs("solution")
        score = self.reward
        filename = f"solution/{self.name}-{score}.txt"
        with open(filename, "w") as f:
            lines = [str(e) for e in self.sol]
            lines += [str(e) for e in set(range(self.D)) - set(self.sol)]
            f.write("\n".join(lines))

    # compute T, R, n_killed from E
    cpdef int score(self):
        cdef int reward = 0
        cdef int enemy_index = 0
        cdef int t, recover_turn, next_enemy, remaining_turns, last_reward_i, rewarding_turns

        self.stamina[:] = 0
        self.stamina[0] = self.Si
        self.n_killed = 0

        # Update T, R and n_killed
        for t in range(0, self.T):
            next_enemy = self.sol[enemy_index]
            if self.stamina[t] > self.Smax:
                self.stamina[t] = self.Smax
            self.stamina[t + 1] += self.stamina[t]
            if self.stamina[t] >= self.Sc_d[next_enemy]:
                enemy_index += 1

                # Update the stamina
                self.stamina[t + 1] -= self.Sc_d[next_enemy]
                recover_turn = t + self.Tr_d[next_enemy]
                if recover_turn < self.T:
                    self.stamina[recover_turn] += self.Sr_d[next_enemy]

                # Compute the reward
                rewarding_turns = self.Nf_d[next_enemy]
                if rewarding_turns > 0:
                    remaining_turns = self.T - t
                    last_reward_i = remaining_turns if remaining_turns < rewarding_turns else rewarding_turns
                    reward += self.Ncum_d[next_enemy][last_reward_i - 1]

        self.n_killed = enemy_index
        self.reward = reward

        return reward

    cpdef simutaed_anealling(self, float temp_i=100, float temp_f=.1, float alpha=0.1, bint verbose=False):
        cdef CProblem p = self
        # initial time
        start = time.time()
        last_time = start

        # find a random solution
        p.sol = np.random.permutation(p.sol)

        cdef float current_temp = temp_i
        cdef int current_score = p.score()
        cdef int[:] current_sol = np.zeros_like(p.sol)
        current_sol[:] = p.sol[:]

        cdef int[:] best_sol = np.zeros_like(p.sol)
        cdef int best_score = current_score
        best_sol[:] = p.sol[:]

        cdef int i = 0
        cdef int index1, index2, score, tmp
        while current_temp > temp_f:
            i += 1
            p.sol[:] = current_sol[:]

            # Swap two random enemies
            if (p.n_killed > 0):
                index_1 = np.random.randint(0, p.n_killed)
            else:
                index_1 = 0
            index_2 = np.random.randint(0, p.sol.size)

            tmp = p.sol[index_1]
            p.sol[index_1] = p.sol[index_2]
            p.sol[index_2] = tmp

            score = p.score()

            # if the new solution is better, accept it. If equal promote diversity
            if score >= current_score:
                current_sol[:] = p.sol[:]
                current_score = score
            # if the new solution is not better, accept it with a probability of e^(-cost/temp)
            elif np.random.uniform(0, 1) < np.exp(-(current_score - score) / current_temp):
                current_sol[:] = p.sol[:]
                current_score = score
                if verbose:
                    print(f"{current_temp}/{temp_f} accepted new solution with lower score {score}")
                    print(f"best_score {best_score}, accepted score {current_score} score {score}")

            if score > best_score:
                best_sol[:] = p.sol[:]
                best_score = score
                if verbose:
                    print(f"{current_temp}/{temp_f} found new best solution {best_score}")
                    print(f"best_score {best_score}, accepted score {current_score} score {score}")

            # report every 10 seconds
            if verbose or (i % 100 == 0 and time.time() - last_time > 10):
                last_time = time.time()
                print(
                    f"{p.name}: {current_temp}/{temp_f} best_score {best_score}, accepted score {current_score}")

            # decrement the temperature
            current_temp -= alpha

        print(f"{p.name}: completed in {time.time() - start} seconds best_score {best_score}")
        p.sol[:] = best_sol[:]
        return p.score()
