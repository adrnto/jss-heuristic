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
                        if op1.machine == op2.machine and op1 in solution.critical_path and op2 in solution.critical_path:  # Solo intercambiar operaciones en la ruta crítica
                            # Si las operaciones están en la misma máquina, intercambiarlas
                            neighbor_solution.O[op1.machine, op1.job], neighbor_solution.O[op2.machine, op2.job] = \
                                neighbor_solution.O[op2.machine, op2.job], neighbor_solution.O[op1.machine, op1.job]

                            # Calcular makespan incrementalmente
                            changed_ops = {neighbor_solution.O[op1.machine, op1.job], neighbor_solution.O[op2.machine, op2.job]}
                            neighbor_solution.makespan = self.get_makespan(neighbor_solution, changed_ops)

                            neighbors.append(neighbor_solution)
                            break  # Salir del bucle interno después de un intercambio
                    else:
                        continue  # Continuar si el bucle interno no se rompe
                    break  # Salir del bucle externo si el bucle interno se rompe

        return neighbors

    def get_makespan(self, solution: Graph, changed_operations=None) -> float:
        """Calculate the makespan, optionally with incremental update."""
        if changed_operations is None:
            calc_tails(solution)  # Full recalculation
        else:
            # Recalculate tails for affected operations and their successors
            affected_operations = set()
            for op in changed_operations:
                affected_operations.add(op)
                affected_operations.update(solution.get_successors(op))
            for op in affected_operations:
                op.tail = 0  # Reset tail
                for succ in solution.get_successors(op):
                    op.tail = max(op.tail, succ.tail + succ.duration)

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
        # Usamos una tupla de movimientos como representación para la lista tabú
        move = self.get_last_move(solution) 
        return move in self.tabu_list

    def get_last_move(self, solution: Graph):
        """Obtener la última operación movida en la solución."""
        # Asumiendo que la solución almacena la información del último movimiento
        # (implementa esto en la función get_neighbors)
        return solution.last_move  

    def update_tabu_list(self, solution: Graph) -> None:
        """Update the tabu list with the new solution.

        Parameters
        ----------
        solution : Graph
            New solution to add to the tabu list
        """
        if len(self.tabu_list) >= self.tabu_size:
            self.tabu_list.pop(0)  # Remove the oldest entry
        move = self.get_last_move(solution)
        self.tabu_list.append(move)

    def iter(self) -> Graph:
        """Iterate to find the best solution using G&T."""
        current_solution = self.build_initial_solution()
        current_cost = self.get_makespan(current_solution)

        for _ in range(self.max_iter):
            neighbors = self.get_neighbors(current_solution)

            # Encontrar el mejor vecino no tabú
            best_neighbor = None
            best_neighbor_cost = np.inf
            for neighbor in neighbors:
                if neighbor.makespan < best_neighbor_cost and not self.is_tabu(neighbor):
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor.makespan

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
