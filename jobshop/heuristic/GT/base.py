import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from jobshop.params import JobShopParams
from jobshop.heuristic.operations import Graph
from jobshop.heuristic.construction import SemiGreedyMakespan, SemiGreedyAlternate
from jobshop.heuristic.evaluation import calc_tails
from jobshop.heuristic.local_search import get_critical, local_search


LARGE_INT = 10000000000000000


class GT:
    def __init__(
        self,
        params: JobShopParams,
        alpha=(0.0, 1.0),
        tabu_size=10,
        max_iter=100,
        seed=None,
    ) -> None:
        """G&T algorithm for the Job-shop scheduling problem

        Parameters
        ----------
        params : JobShopParams
            Problem parameters

        tabu_size : int, optional
            Size of the tabu list, by default 10

        max_iter : int, optional
            Maximum number of iterations, by default 1000

        seed : int | None, optional
            numpy random seed, by default None
        """
        self.params = params
        self.tabu_size = tabu_size
        self.max_iter = max_iter
        self.seed = seed
        self.tabu_list = []
        self.best_solution = None
        self.best_cost = np.inf
        self.construction = SemiGreedyMakespan(alpha)

    def build_initial_solution(self) -> Graph:
        """Build an initial solution using a greedy approach.

        Returns
        -------
        Graph
            Initial solution
        """
        S = Graph(self.params)
        S = self.construction(S)
        
        calc_tails(S)
        get_critical(S)
        S = local_search(S)
        return S

    def get_neighbors(self, solution: Graph) -> list:
        """Generate neighbors for the given solution.

        Parameters
        ----------
        solution : Graph
            Current solution

        Returns
        -------
        list
            List of neighboring solutions
        """
        neighbors = []

        # Obtener la lista de trabajos (asumiendo que las claves de O son tuplas (máquina, trabajo))
        jobs = set(job for _, job in solution.O.keys())

        # Iterar sobre todos los trabajos para generar vecinos
        job_list = list(jobs)
        for i in range(len(job_list)):
            for j in range(i + 1, len(job_list)):
                job1 = job_list[i]
                job2 = job_list[j]

                # Crear una copia de la solución actual
                neighbor_solution = solution.copy()

                # Buscar las operaciones de job1 y job2 en las máquinas
                op1_list = [neighbor_solution.O[machine, job1] for machine, job in neighbor_solution.O.keys() if job == job1]
                op2_list = [neighbor_solution.O[machine, job2] for machine, job in neighbor_solution.O.keys() if job == job2]

                # Verificar si ambos trabajos tienen operaciones en la misma máquina
                for op1 in op1_list:
                    for op2 in op2_list:
                        if op1.machine == op2.machine:
                            # Si las operaciones están en la misma máquina, intercambiarlas
                            neighbor_solution.O[op1.machine, op1.job], neighbor_solution.O[op2.machine, op2.job] = \
                                neighbor_solution.O[op2.machine, op2.job], neighbor_solution.O[op1.machine, op1.job]

                # Agregar la solución vecina a la lista
                neighbors.append(neighbor_solution)

        return neighbors
        
    def evaluate_makespan_parallel(self, neighbors):
        """Evaluar los makespans de los vecinos en paralelo."""
        with ThreadPoolExecutor() as executor:
            # `executor.map` ejecuta `self.get_makespan` para cada vecino en paralelo
            costs = list(executor.map(self.get_makespan, neighbors))
        return costs
    
    
    def get_makespan(self, solution: Graph) -> float:
        """Calculate the makespan of the given solution.

        Parameters
        ----------
        solution : Graph
            Solution to evaluate

        Returns
        -------
        float
            Makespan of the solution
        """
        # The makespan is the total time required to complete all jobs
        calc_tails(solution)  # Calculate tail times for each operation
        return max(op.release + op.duration for op in solution.O.values())

    def is_tabu(self, solution: Graph) -> bool:
        """Check if the solution is in the tabu list.

        Parameters
        ----------
        solution : Graph
            Solution to check

        Returns
        -------
        bool
            True if the solution is tabu, False otherwise
        """
        return solution.signature in self.tabu_list

    def update_tabu_list(self, solution: Graph) -> None:
        """Update the tabu list with the new solution.

        Parameters
        ----------
        solution : Graph
            New solution to add to the tabu list
        """
        if len(self.tabu_list) >= self.tabu_size:
            self.tabu_list.pop(0)  # Remove the oldest entry
        self.tabu_list.append(solution.signature)

    def iter(self) -> Graph:
        """Iterate to find the best solution using G&T."""
        current_solution = self.build_initial_solution()
        current_cost = self.get_makespan(current_solution)

        for _ in range(self.max_iter):
            neighbors = self.get_neighbors(current_solution)

            # Usamos `evaluate_makespan_parallel` para calcular los makespans de los vecinos en paralelo
            neighbor_costs = self.evaluate_makespan_parallel(neighbors)

            best_neighbor = None
            best_neighbor_cost = np.inf

            # Encontrar el mejor vecino no tabú
            for i, neighbor in enumerate(neighbors):
                neighbor_cost = neighbor_costs[i]  # Usamos el costo calculado previamente en paralelo
                if neighbor_cost < best_neighbor_cost and not self.is_tabu(neighbor):
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost

            # Si se encuentra un vecino no tabú, moverse a esa solución
            if best_neighbor is not None:
                current_solution = best_neighbor
                current_cost = best_neighbor_cost
                self.update_tabu_list(current_solution)

                # Actualizar la mejor solución encontrada
                if current_cost < self.best_cost:
                    self.best_solution = current_solution
                    self.best_cost = current_cost

        return self.best_solution

    def __call__(self) -> Graph:
        """Run the G&T algorithm to solve the Job-shop scheduling problem.

        Returns
        -------
        Graph
            Best solution found
        """
        return self.iter()
