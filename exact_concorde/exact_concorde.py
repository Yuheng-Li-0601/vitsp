from concorde.tsp import TSPSolver
from concorde.tests.data_utils import get_dataset_path
import time
import tempfile
import os
import numpy as np

class Concorde:
    def __init__(self, nodes=None, coordinates=None, dist_matrix=None, file_path=None):
        """
        Initialize the TSPInstance. Accepts either nodes with coordinates or a distance matrix.
        """
        self.nodes = nodes
        self.coordinates = coordinates
        self.dist_matrix = dist_matrix
        self.solution = None
        self.route = []
        self.obj_value = 0
        self.latency = 0

        if not ((self.nodes and self.coordinates) or self.dist_matrix is not None):
            raise ValueError("Either (nodes and coordinates) or a distance matrix must be provided.")

        # Generate a pseudo TSP file and initialize the solver if instance is generated on the go
        # if the instance has a real file, directly import this file
        if file_path is None:
            self.pseudo_file = self._generate_pseudo_file()
            self.solver = TSPSolver.from_tspfile(self.pseudo_file.name)
        else:
            self.solver = TSPSolver.from_tspfile(file_path)

    def _generate_pseudo_file(self):
        """
        Dynamically generate a .tsp file with either nodes and coordinates or a distance matrix.
        Returns a temporary file object.
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tsp", mode="w")

        temp_file.write(f"NAME : Pseudo_TSP_Instance\n")
        temp_file.write(f"TYPE: TSP\n")

        if self.dist_matrix is not None:
            dimension = len(self.dist_matrix)
            temp_file.write(f"DIMENSION : {dimension}\n")
            temp_file.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
            temp_file.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n") # TODO: use UPPER_ROW to save time
            temp_file.write("EDGE_WEIGHT_SECTION\n")

            for row in self.dist_matrix:
                temp_file.write(" ".join(str(int(dist)) for dist in row) + "\n")

        elif self.nodes and self.coordinates:
            dimension = len(self.nodes)
            temp_file.write(f"DIMENSION : {dimension}\n")
            temp_file.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
            temp_file.write("NODE_COORD_SECTION\n")

            for i, (x, y) in enumerate(self.coordinates, start=1):
                temp_file.write(f"{i} {x} {y}\n")

        else:
            raise ValueError("Invalid input: Either (nodes and coordinates) or a distance matrix must be provided.")

        temp_file.write("EOF\n")
        temp_file.flush()  # Ensure all data is written to disk
        return temp_file

    def optimize(self, timelimit: float = -1.0, verbose=True):
        start_time = time.time()
        self.solution = self.solver.solve(time_bound=timelimit, verbose=verbose)
        self.latency = time.time() - start_time

        self.route = self.solution.tour.tolist()
        self.route.append(self.route[0])  #TODO: check if need

    def get_tsp_route(self):
        return self.route
    def get_objective_value(self):
        return self.solution.optimal_value

    def cleanup(self):
        """
        Clean up the temporary pseudo TSP file.
        """
        if os.path.exists(self.pseudo_file.name):
            os.unlink(self.pseudo_file.name)

def determine_instance_boundary(coordinates):
    MARGIN = 0
    x_min = min(coord[0] for coord in coordinates) - MARGIN
    x_max = max(coord[0] for coord in coordinates) + MARGIN
    y_min = min(coord[1] for coord in coordinates) - MARGIN
    y_max = max(coord[1] for coord in coordinates) + MARGIN

    grid_resolution = 1000 if max((x_max-x_min), (y_max-y_min)) > 5000 else 100

    return int(x_min), int(x_max), int(y_min), int(y_max), int(grid_resolution)


if __name__ == '__main__':
    from .helper.parse_instances import FileParser
    from .helper.plot_solution import SolutionPlot
    from .heuristic_tsp.tsp import TravelingSalesmenProblem

    file_parser = FileParser()
    solution_plotter = SolutionPlot()

    fname = './instances/tsplib/a280.tsp'

    instance_info = file_parser.parse_instance_from_file(fname)
    coordinates = instance_info['COORDINATES']
    distance_mat = np.array(instance_info['COST_MATRIX'])
    nodes = {i: (x, y) for i, (x, y) in enumerate(coordinates)}
    tsp_instance = TravelingSalesmenProblem(node_coords_dict=nodes)

    concorde_model = Concorde(nodes=list(nodes.keys()), coordinates=coordinates)
    concorde_model.optimize(timelimit=100, verbose=False)
    current_route = concorde_model.get_tsp_route()
    current_obj = concorde_model.get_objective_value()

    X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES = determine_instance_boundary(coordinates)

    tsp_plot = solution_plotter.plot_tsp_solution(current_route, tsp_instance.coords,
                                                          x_min=X_MIN, x_max=X_MAX,
                                                          y_min=Y_MIN, y_max=Y_MAX,
                                                          grid_resolution=(X_MAX - X_MIN)//10)