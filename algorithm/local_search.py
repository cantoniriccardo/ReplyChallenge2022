import numpy as np
from algorithm.random_solution import iterated_random
import time

def iterated_local_search(solution, n_iter=1000):
    # find a random solution
    iterated_random(solution, n_iter=100)

    best_sol = np.zeros_like(solution.sol)
    np.copyto(best_sol, solution.sol)
    best_score = solution.score()

    for i in range(n_iter):
        np.random.shuffle(solution.sol)

        # Swap two random enemies
        index_1 = np.random.randint(0, solution.sol.size)
        index_2 = np.random.randint(0, solution.sol.size)

        tmp = solution.sol[index_1]
        solution.sol[index_1] = solution.sol[index_2]
        solution.sol[index_2] = tmp

        # Score
        score = solution.score()

        if score > best_score:
            np.copyto(best_sol, solution.sol)
            best_score = score
            print(f"{i}/{n_iter} found new best solution {best_score}")

    solution.sol = best_sol
    return solution.score()


def simulated_annealing(solution, temp_i=10000, temp_f=.1, alpha=10, verbose=False):
    # initial time
    start = time.time()
    last_time = start

    # find a random solution
    iterated_random(solution, n_iter=100)

    current_temp = temp_i
    current_score = solution.score()
    current_sol = np.zeros_like(solution.sol)
    np.copyto(current_sol, solution.sol)

    best_sol = np.zeros_like(solution.sol)
    best_score = current_score
    np.copyto(best_sol, solution.sol)

    i = 0
    while current_temp > temp_f:
        i += 1
        np.copyto(solution.sol, current_sol)

        # Swap two random enemies
        if(solution.n_killed > 0):
            index_1 = np.random.randint(0, solution.n_killed)
        else:
            index_1 = 0
        index_2 = np.random.randint(0, solution.sol.size)

        tmp = solution.sol[index_1]
        solution.sol[index_1] = solution.sol[index_2]
        solution.sol[index_2] = tmp

        score = solution.score()

        # if the new solution is better, accept it. If equal promote diversity
        if score >= current_score:
            np.copyto(current_sol, solution.sol)
            current_score = score
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        elif np.random.uniform(0, 1) < np.exp(-(current_score - score) / current_temp):
            np.copyto(current_sol, solution.sol)
            current_score = score
            if verbose:
                print(f"{current_temp}/{temp_f} accepted new solution with lower score {score}")
                print(f"best_score {best_score}, accepted score {current_score} score {score}")

        if score > best_score:
            np.copyto(best_sol, solution.sol)
            best_score = score
            if verbose:
                print(f"{current_temp}/{temp_f} found new best solution {best_score}")
                print(f"best_score {best_score}, accepted score {current_score} score {score}")

        # report every 10 seconds
        if verbose or (i % 100 == 0 and time.time() - last_time > 10):
            last_time = time.time()
            print(f"{solution.problem.name}: {current_temp}/{temp_f} best_score {best_score}, accepted score {current_score}")

        # decrement the temperature
        current_temp -= alpha

    print(f"{solution.problem.name}: completed in {time.time() - start} seconds best_score {best_score}")
    np.copyto(solution.sol, best_sol)
    return solution.score()
