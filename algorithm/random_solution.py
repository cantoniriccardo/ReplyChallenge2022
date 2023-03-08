import numpy as np


def random_solution(solution):
    np.random.shuffle(solution.sol)
    solution.score()


def iterated_random(solution, n_iter=1000):
    best_sol = np.array(range(len(solution.sol)))
    best_score = 0

    for i in range(n_iter):
        np.random.shuffle(solution.sol)
        score = solution.score()

        if score > best_score:
            np.copyto(best_sol, solution.sol)
            best_score = score
            print(f"{i}/{n_iter} found new best solution {best_score}")

    solution.sol = best_sol
    return solution.score()
