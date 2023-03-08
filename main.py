import os
from algorithm.local_search import simulated_annealing
from problem import Problem
from cproblem import CProblem
from solution import Solution
from multiprocessing import Pool


def solve_instance(filename):
    problem = Problem.from_file(filename)
    solution = Solution(problem)
    simulated_annealing(solution, temp_i=200, temp_f=.1, alpha=0.005, verbose=False)
    solution.dump()


def csolve_instance(filename):
    problem = CProblem.from_file(filename)
    problem.simutaed_anealling(temp_i=50, temp_f=.1, alpha=0.00005, verbose=False)
    problem.dump()


if __name__ == "__main__":
    with Pool(6) as p:
        p.map(csolve_instance, [f"data/{f}" for f in os.listdir("data") if f.endswith(".txt")])