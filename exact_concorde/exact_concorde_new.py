import subprocess
import os
import time
import tempfile
import numpy as np

class Concorde:
    def __init__(self, nodes=None, coordinates=None, dist_matrix=None, file_path=None):
        self.nodes = nodes
        self.coordinates = coordinates
        self.dist_matrix = dist_matrix
        self.solution_route = None
        self.route = []
        self.obj_value = 0
        self.latency = 0
        
        # point to the saving position of concorde in vitsp/solver_bin
        self.bin_path = "/workspace/codes/vitsp/solver_bin/concorde"

        if not ((self.nodes and self.coordinates) or self.dist_matrix is not None or file_path):
            raise ValueError("Invalid input for Concorde.")
        
        self.tsp_file_path = file_path if file_path else self._generate_pseudo_file()

    def _generate_pseudo_file(self):
        # Each Concorde invocation gets its own temp directory to avoid
        # intermediate-file conflicts when running multiple solvers in parallel.
        self._tmpdir = tempfile.mkdtemp(prefix="concorde_run_")
        fd, path = tempfile.mkstemp(suffix=".tsp", prefix="concorde_sub_", dir=self._tmpdir)
        with os.fdopen(fd, 'w') as f:
            f.write(f"NAME : Pseudo_TSP\nTYPE: TSP\n")
            if self.dist_matrix is not None:
                dimension = len(self.dist_matrix)
                f.write(f"DIMENSION : {dimension}\nEDGE_WEIGHT_TYPE: EXPLICIT\n")
                f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")

                
                for row in self.dist_matrix:
                    f.write(" ".join(str(int(dist)) for dist in row) + "\n")
            else:
                dimension = len(self.nodes)
                f.write(f"DIMENSION : {dimension}\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
                for i, (x, y) in enumerate(self.coordinates, start=1):
                    f.write(f"{i} {x} {y}\n")
            f.write("EOF\n")
        return path

    def optimize(self, timelimit: float = -1.0, verbose=False):
        start_time = time.time()
        # Use absolute path so the solution file is found regardless of
        # the Python process's working directory.
        work_dir = getattr(self, '_tmpdir', os.path.dirname(self.tsp_file_path))
        sol_file = self.tsp_file_path.replace(".tsp", ".sol")
        
        try:
            timeout_val = timelimit if timelimit > 0 else None
            subprocess.run([self.bin_path, "-o", sol_file, self.tsp_file_path], 
                           stdout=subprocess.DEVNULL if not verbose else None,
                           stderr=subprocess.PIPE,
                           check=True,
                           timeout=timeout_val,
                           cwd=work_dir)
            
            if os.path.exists(sol_file):
                with open(sol_file, "r") as f:
                    data = f.read().split()
                    self.solution_route = [int(x) for x in data[1:]]
                    self.route = self.solution_route.copy()
            else:
                print(f"[WARNING] Concorde solution file not found: {sol_file}")
            
            self._calculate_obj()
            
        except Exception as e:
            print(f"[ERROR] Concorde Binary failed: {e}")
        finally:
            self.latency = time.time() - start_time
            self._cleanup(sol_file)

    def _calculate_obj(self):
        # calculate the objective value even without library, for solver_master
        if self.dist_matrix is not None and self.solution_route:
            obj = 0
            n = len(self.solution_route)
            for i in range(n):
                u = self.solution_route[i]
                v = self.solution_route[(i + 1) % n]
                obj += self.dist_matrix[u][v]
            self.obj_value = obj
        elif self.coordinates is not None and self.solution_route:
            # calculate the distance between nodes
            obj = 0
            coords = np.array(self.coordinates)
            n = len(self.solution_route)
            for i in range(n):
                u, v = self.solution_route[i], self.solution_route[(i+1)%n]
                obj += np.linalg.norm(coords[u] - coords[v])
            self.obj_value = obj

    def get_tsp_route(self):
        return self.route
    
    def get_objective_value(self):
        return self.obj_value

    def _cleanup(self, sol_file):
        for ext in [".res", ".pul", ".sav", ".mas", ".sol"]:
            f_to_del = self.tsp_file_path.replace(".tsp", ext)
            if os.path.exists(f_to_del): os.remove(f_to_del)
        if os.path.exists(self.tsp_file_path): os.remove(self.tsp_file_path)
        # Remove the per-invocation temp directory if it was created
        work_dir = getattr(self, '_tmpdir', None)
        if work_dir and os.path.isdir(work_dir):
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)

def determine_instance_boundary(coordinates):
    MARGIN = 0  
    
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]
    
    x_min, x_max = min(x_coords) - MARGIN, max(x_coords) + MARGIN
    y_min, y_max = min(y_coords) - MARGIN, max(y_coords) + MARGIN

    grid_resolution = 1000 if max((x_max - x_min), (y_max - y_min)) > 5000 else 100

    return int(x_min), int(x_max), int(y_min), int(y_max), int(grid_resolution)

if __name__ == '__main__':
    print("[TEST] Generating the random test data and set up Concorde...")
    
    # 1.randomly generate test data
    num_test_nodes = 20
    # scale: [0, 1000]
    test_coords = np.random.randint(0, 1000, size=(num_test_nodes, 2)).tolist()
    test_nodes = list(range(num_test_nodes))

    print(f"-> Successfully generated {num_test_nodes} random nodes.")

    # 2. initialize
    try:
        concorde_model = Concorde(nodes=test_nodes, coordinates=test_coords)
        
        # report time
        concorde_model.optimize(timelimit=10, verbose=True)
        
        current_route = concorde_model.get_tsp_route()
        current_obj = concorde_model.get_objective_value()

        # 3. test results
        print("\n" + "="*30)
        print("[SUCCESS] Concorde succeeds！")
        print(f"-> Solved! Totally get the {len(current_route)} nodes route.")
        print(f"-> Total Distance: {round(current_obj, 2)}")
        print(f"-> latency: {round(concorde_model.latency, 4)}s")
        
        # 4. test boundary function
        X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES = determine_instance_boundary(test_coords)
        print(f"-> (MARGIN={10}): X[{X_MIN}, {X_MAX}], Y[{Y_MIN}, {Y_MAX}]")
        print("="*30)

    except Exception as e:
        print(f"\n[ERROR] error: {e}")
        print("please test whether /workspace/codes/vitsp/solver_bin/concorde exists and is executable.")