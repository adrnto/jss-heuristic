import numpy as np
from jobshop.params import JobShopParams
from jobshop.heuristic.operations import Graph
from jobshop.heuristic.evaluation import calc_tails
from jobshop.heuristic.local_search import get_critical, local_search


class GifflerThompson:
    
    def __init__(self, params: JobShopParams) -> None:
        """Giffler & Thompson algorithm for Job-shop scheduling problem
        
        Parameters
        ----------
        params : JobShopParams
            Problem parameters
        """
        self.params = params

    def get_feasible_operations(self, S: Graph):
        """Obtiene las operaciones factibles (no bloqueadas por restricciones de precedencia)

        Parameters
        ----------
        S : Graph
            La solución parcial actual

        Returns
        -------
        List
            Lista de operaciones factibles (no bloqueadas)
        """
        feasible_ops = []
        for job in S.jobs:
            # Obtenemos la primera operación no programada de cada trabajo
            if job.next_operation is not None:
                feasible_ops.append(job.next_operation)
        return feasible_ops

    def select_operation(self, feasible_ops, S: Graph):
        """Selecciona la operación a programar basada en el menor tiempo de inicio posible (regla de prioridad)

        Parameters
        ----------
        feasible_ops : list
            Lista de operaciones factibles (no bloqueadas)
        S : Graph
            La solución parcial actual

        Returns
        -------
        operation
            La operación seleccionada para ser programada
        """
        # Encontrar la operación con el menor tiempo de inicio posible en la máquina
        selected_op = None
        min_time = float('inf')

        for op in feasible_ops:
            start_time = max(S.machine_avail[op.machine], S.job_avail[op.job])
            if start_time < min_time:
                min_time = start_time
                selected_op = op

        return selected_op

    def build_solution(self) -> Graph:
        """Construye una solución usando el enfoque secuencial del algoritmo Giffler & Thompson

        Returns
        -------
        Graph
            Solución
        """
        # Inicializar una solución vacía
        S = Graph(self.params)

        # Inicializar las disponibilidades de máquinas y trabajos
        S.machine_avail = np.zeros(len(self.params.machines))
        S.job_avail = np.zeros(len(self.params.jobs))

        while not S.is_complete():
            # Paso 1: Obtener todas las operaciones factibles
            feasible_ops = self.get_feasible_operations(S)
            
            # Paso 2: Seleccionar la operación que minimiza el tiempo de inicio
            selected_op = self.select_operation(feasible_ops, S)
            
            # Paso 3: Programar la operación seleccionada
            start_time = max(S.machine_avail[selected_op.machine], S.job_avail[selected_op.job])
            selected_op.start = start_time
            selected_op.end = start_time + selected_op.duration

            # Actualizar disponibilidades de la máquina y el trabajo
            S.machine_avail[selected_op.machine] = selected_op.end
            S.job_avail[selected_op.job] = selected_op.end

            # Marcar la operación como completada
            S.schedule_operation(selected_op)

        # Calcular tiempos finales y operaciones críticas
        calc_tails(S)
        get_critical(S)
        S = local_search(S)  # Aplicar búsqueda local (opcional)

        return S

    def __call__(self, verbose=False) -> Graph:
        """Ejecutar el algoritmo Giffler & Thompson

        Parameters
        ----------
        verbose : bool, optional
            Si se debe imprimir la salida, por defecto False

        Returns
        -------
        Graph
            Solución final
        """
        S_best = self.build_solution()
        if verbose:
            print(f"Solución construida con G&T - Makespan: {S_best.C}")
        return S_best

