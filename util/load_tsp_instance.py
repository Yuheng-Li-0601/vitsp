
import requests, tarfile, os, gzip, shutil
from tqdm.auto import tqdm
from tsplib95.loaders import load_problem, load_solution
import torch
import tsplib95
from tensordict import TensorDict

def download_and_extract_tsplib(url, directory="atsplib", delete_after_unzip=True):
    os.makedirs(directory, exist_ok=True)

    # Download with progress bar
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open("tsplib.tar.gz", 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                pbar.update(len(chunk))

    # Extract tar.gz
    # Try the following, either works
    # with tarfile.open("tsplib.tar.gz", 'r:gz') as tar:
    #     tar.extractall(directory)
    #
    with tarfile.open("tsplib.tar.gz", 'r') as tar:
        tar.extractall(directory)

    # Decompress .gz files inside directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".gz"):
                path = os.path.join(root, file)
                with gzip.open(path, 'rb') as f_in, open(path[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(path)

    if delete_after_unzip:
        os.remove("tsplib.tar.gz")

def normalize_coord(coord:torch.Tensor) -> torch.Tensor:
    x, y = coord[:, 0], coord[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_scaled = (x - x_min) / (x_max - x_min)
    y_scaled = (y - y_min) / (y_max - y_min)
    coord_scaled = torch.stack([x_scaled, y_scaled], dim=1)
    return coord_scaled

def load_tsp_instance(tsplib_dir, torch_tensor=True, normalize=True):
    files = os.listdir(tsplib_dir)
    problem_files_full = [file for file in files if file.endswith('.tsp')]

    # Load the optimal solution files from TSPLib
    solution_files = [file for file in files if file.endswith('.opt.tour')]

    problems = []
    for problem_file in problem_files_full:
        problem = tsplib95.load(os.path.join(tsplib_dir, problem_file))

        if not len(problem.node_coords):
            continue

        sol_file = problem_file.replace('.tsp', '.opt.tour')

        if torch_tensor:
            node_coords = torch.tensor([v for v in problem.node_coords.values()])
        else:
            node_coords = [v for v in problem.node_coords.values()]

        if sol_file in solution_files:
            solution = tsplib95.load(os.path.join(tsplib_dir, sol_file))
        else:
            solution = None
        problems.append({
            "name": sol_file.replace('.opt.tour', ''),
            "node_coords": node_coords,
            "solution":  solution.tours[0] if solution else [],
            "dimension": problem.dimension
        })

    # order by dimension
    problems = sorted(problems, key=lambda x: x['dimension'])

    return problems

def load_single_tsp_instance(tsplib_dir, torch_tensor=True, normalize=True):

    def normalize_coord(coords, x_min, x_max, y_min, y_max, scale=1000):
        return [
            (
                (x - x_min) / (x_max - x_min) * scale,
                (y - y_min) / (y_max - y_min) * scale
            )
            for x, y in coords
        ]

    def determine_instance_boundary(coordinates):
        MARGIN = 0
        x_min = min(coord[0] for coord in coordinates) - MARGIN
        x_max = max(coord[0] for coord in coordinates) + MARGIN
        y_min = min(coord[1] for coord in coordinates) - MARGIN
        y_max = max(coord[1] for coord in coordinates) + MARGIN

        grid_resolution = 1000 if max((x_max - x_min), (y_max - y_min)) > 5000 else 100

        return x_min, x_max, y_min, y_max, grid_resolution
    # tsplib_dir mush end up with tsp

    problems = []
    problem = tsplib95.load(tsplib_dir)

    if normalize:
        X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES = determine_instance_boundary(list(problem.node_coords.values()))
        norm_node_coords = normalize_coord(list(problem.node_coords.values()),  X_MIN, X_MAX, Y_MIN, Y_MAX, 1)

        if torch_tensor:
            node_coords = torch.tensor([v for v in norm_node_coords])
        else:
            node_coords = [v for v in norm_node_coords]

    else:
        if torch_tensor:
            node_coords = torch.tensor([v for v in problem.node_coords.values()])
        else:
            node_coords = [v for v in problem.node_coords.values()]

    problems.append({
        "node_coords": node_coords,
        "solution":  [],
        "dimension": problem.dimension
    })

    return problems

def tsplib_to_td(problem, normalize=True):
    coords = torch.tensor(problem['node_coords']).float()
    coords_norm = normalize_coord(coords) if normalize else coords
    td = TensorDict({
        'locs': coords_norm,
    })
    td = td[None] # add batch dimension, in this case just 1
    return td

if __name__ == "__main__":
    # download_and_extract_tsplib("http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz")
    download_and_extract_tsplib('http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/atsp/ALL_atsp.tar')
