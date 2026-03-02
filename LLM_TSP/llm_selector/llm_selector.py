from dataclasses import dataclass
import logging
import psutil
from typing import Iterable, List, Tuple, Union, Any, Dict
from multiprocessing import Queue
import queue
from helper.parse_llm_response import LLMTextParser
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import random

@dataclass
class Subproblem:
    removed_nodes_list: list
    route_segments_list: list
    coordinates_list: list
    current_obj: float
    solution_version: float
    llm_source: any


def get_all_from_queue(q: Queue):
    items = []
    while True:
        try:
            item = q.get_nowait()
            items.append(item)
        except queue.Empty:
            break
    return items

def set_cpu_affinity(process, cpu_ids):
    """
    Set CPU affinity for a process.
    """
    p = psutil.Process(process.pid)
    p.cpu_affinity(cpu_ids)  # Assign specific CPUs to the process

def assign_proportional_cores_to_tasks(tasks: List[Any]) -> Dict[int, List[int]]:
    """
    Proportionally assign available CPU cores to the given list of tasks.
    """
    total_cores = os.cpu_count() or 1
    total_task_lengths = sum(len(task[0]) for task in tasks)

    core_assignment = {}
    current_core = 0

    for i, task in enumerate(tasks):
        weight = len(task[0]) / total_task_lengths if total_task_lengths > 0 else 1 / len(tasks)
        allocated_cores = max(1, int(round(weight * total_cores)))
        core_assignment[i] = list(range(current_core, min(current_core + allocated_cores, total_cores)))
        current_core += allocated_cores
        if current_core >= total_cores:
            break

    return core_assignment

def parse_num_tag(text: str, as_float: bool = False) -> Union[int, float]:
    """
    Return the numeric value inside a <num> … </num> tag.

    Parameters
    ----------
    text : str
        The string that contains a single <num> … </num> element.
    as_float : bool, default False
        • False  → return an int if the content looks like an integer.
        • True   → always return a float.

    Raises
    ------
    ValueError
        If no <num> tag is present or the content is not a valid number.
    """
    m = _NUM_RE.search(text)
    if not m:
        raise ValueError("No <num> … </num> tag with a valid number found")

    number_str = m.group(1)
    return float(number_str) if as_float or "." in number_str else int(number_str)

def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(processName)s] %(message)s",
                        datefmt="%H:%M:%S",
                        force=True)

def llm_visual_selection(args, solution_plotter, current_route, pending_subproblem_queue, tsp_instance, prior_selection_traj,
                         llm_selector, backup_selector, llm_parser, num_subregion,
                         x_min, x_max, y_min, y_max, grid_res, traj_lock, num_nodes):

    #TODO: optimize the plot generation using GPU?
    tsp_plot = solution_plotter.plot_tsp_solution_plotly(args, current_route, tsp_instance.coords,
                                                         x_min=int(x_min), x_max=int(x_max),
                                                         y_min=int(y_min), y_max=int(y_max),
                                                         grid_resolution=int(grid_res), node_size=2, num_nodes=num_nodes)

    #TODO: input the buffer area to avoid selection
    current_selector = llm_selector.get_next_llm()

    pending_subproblems = peek_queue(pending_subproblem_queue, traj_lock)
    prior_selections = peek_queue(prior_selection_traj, traj_lock)

    pending_coords = [coord for s_p in pending_subproblems for coord in s_p.coordinates_list]
    print("Pending length is ", pending_coords)

    response = current_selector.vision_chat(fig=tsp_plot,
                                            prior_selection=prior_selections,
                                            num_region=num_subregion,
                                            pending_coords=pending_coords,
                                            x_min=int(x_min),
                                            x_max=int(x_max),
                                            y_min=int(y_min),
                                            y_max=int(y_max)
                                            )

    coordinates_list = llm_parser.parse_subrectangle_coordinates(response)

    if coordinates_list[0][0] is None:
        coordinates_list = backup_selector.generate_subrectangle(X_MIN=x_min,
                                                                 X_MAX=x_max,
                                                                 Y_MIN=y_min,
                                                                 Y_MAX=y_max,
                                                                 no_more_than=grid_res,
                                                                 num_region=num_subregion)



    return coordinates_list

def handle_valid_subproblem_selection(args, removed_nodes, coordinates_list, solution_plotter, current_route,
                                            pending_subproblem_queue, tsp_instance, prior_selection, llm_selector, backup_selector,
                                            llm_parser, num_subregion,
                                            current_route_edges, X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES, traj_lock):

    while len(removed_nodes) < 5 or len(removed_nodes) > args.max_node_for_solver:

        if len(removed_nodes) < 5:
            coordinates_list = backup_selector.generate_subrectangle(X_MIN=X_MIN,
                                                                     X_MAX=X_MAX,
                                                                     Y_MIN=Y_MIN,
                                                                     Y_MAX=Y_MAX,
                                                                     num_region=1
                                                                     )
            x_min, x_max, y_min, y_max = coordinates_list[0]

        elif len(removed_nodes) > args.max_node_for_solver:
            x_min, x_max, y_min, y_max = coordinates_list[0]
            print(f'coordinate list is {x_min, x_max, y_min, y_max}')

            grid_resolution = max((x_max - x_min), (y_max - y_min)) // args.gridding_resolution

            if args.fast_llm_model == 'random':
                coordinates_list = backup_selector.generate_subrectangle(X_MIN=x_min,
                                                                         X_MAX=x_max,
                                                                         Y_MIN=y_min,
                                                                         Y_MAX=y_max,
                                                                         num_region=1
                                                                         )
                time.sleep(5) # minimicing the arrival rate of OpenAI response
            else:
                # select a region from zoomed in region
                coordinates_list = llm_visual_selection(args, solution_plotter, current_route, pending_subproblem_queue,
                                                        tsp_instance,
                                                        prior_selection, llm_selector, backup_selector,
                                                        llm_parser, 1,
                                                        x_min, x_max, y_min, y_max, grid_resolution, traj_lock,
                                                        num_nodes = len(removed_nodes),)

            print(f'zoomed in coordinate list is {coordinates_list}')

        updated_route_edges, removed_nodes, route_segments = tsp_instance.remove_edges_in_subrectangle(current_route,
                                                                                                       current_route_edges,
                                                                                                       coordinates_list)

    return route_segments, removed_nodes, coordinates_list

def process_coordinate(coordinate, args, tsp_instance, current_route, current_route_edges,
                        pending_subproblem_queue, coordinates_list, solution_plotter, prior_selection,
                        llm_selector, backup_selector,
                        llm_parser, num_subregion, X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES, traj_lock):

    coordinate = [coordinate]
    print(coordinate)

    updated_route_edges, removed_nodes, route_segments = tsp_instance.remove_edges_in_subrectangle(
        current_route, current_route_edges, coordinate
    )
    origin_removed_nodes = removed_nodes
    if (len(removed_nodes) == 0) or (len(removed_nodes) > args.max_node_for_solver):
        route_segments, removed_nodes, coordinate = handle_valid_subproblem_selection(
            args, removed_nodes, coordinate, solution_plotter, current_route, pending_subproblem_queue,
            tsp_instance, prior_selection, llm_selector,
            backup_selector, llm_parser, args.llm_subproblem_selection, current_route_edges,
            X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES, traj_lock
        )

    print(f'original removed nodes is {len(origin_removed_nodes)}')
    print(f'new removed nodes is {len(removed_nodes)}')
    print(f'overlapping with the original one: {len(set(origin_removed_nodes).intersection(set(removed_nodes)))}')

    return removed_nodes, route_segments, coordinate

def llm_io(args, solution_plotter, current_route, current_route_edges, pending_subproblem_queue, tsp_instance, prior_selection, llm_selector,
                 backup_selector, llm_parser, num_subregion,
                 x_min, x_max, y_min, y_max, grid_res, num_nodes, traj_lock) -> List[int]:

    # await asyncio.sleep(random.uniform(10, 20))
    # await asyncio.sleep(latency)

    # --- Obtain subproblem selection
    # if True:
    if args.select_sequence:
        removed_nodes_list = []
        route_segments_list = []
        zoomed_in_coordinates_list = []
       
        for i in range(args.llm_subproblem_selection):
            removed_nodes = backup_selector.generate_sequence(current_route, randomize=args.random_selection, max_length=args.max_node_for_solver)
            removed_nodes_list.append(removed_nodes)
        time.sleep(5)
        
    else:

        if args.hard_coded_subrectangle:
            coordinates_list = [(8408, 8927, 5921, 9508), (7375, 8207, 7090, 9756)]
            time.sleep(5)
        elif args.fast_llm_model == 'random':
            coordinates_list = backup_selector.generate_subrectangle(X_MIN=x_min,
                                                                    X_MAX=x_max,
                                                                    Y_MIN=y_min,
                                                                    Y_MAX=y_max,
                                                                    no_more_than=grid_res,
                                                                    num_region=args.llm_subproblem_selection)
            time.sleep(5)

        else:
            coordinates_list = llm_visual_selection(args, solution_plotter, current_route, pending_subproblem_queue, tsp_instance, prior_selection,
                                llm_selector, backup_selector, llm_parser, args.llm_subproblem_selection,
                                x_min, x_max, y_min, y_max, grid_res, traj_lock, len(current_route), )


        # ---- Process the selection to ensure the selection is valid (does not cover too large area)
        removed_nodes_list = []
        route_segments_list = []
        zoomed_in_coordinates_list = []

        # Prepare a partial function to pass shared arguments
        process_func = partial(
            process_coordinate, args=args, tsp_instance=tsp_instance, current_route=current_route,
            current_route_edges=current_route_edges, pending_subproblem_queue=pending_subproblem_queue,
            coordinates_list=coordinates_list,
            solution_plotter=solution_plotter, prior_selection=prior_selection,
            llm_selector=llm_selector, backup_selector=backup_selector,
            llm_parser=llm_parser, num_subregion=args.llm_subproblem_selection, X_MIN=x_min, X_MAX=x_max,
            Y_MIN=y_min, Y_MAX=y_max, GRID_RES=grid_res, traj_lock=traj_lock
        )

        num_thread = len(coordinates_list)

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=num_thread) as executor:
            results = list(executor.map(process_func,
                                        coordinates_list))  # elements within coordinates_list will be split dynamically to threads

        # Unpack results
        for removed_nodes, route_segments, coordinates_list in results:
            removed_nodes_list.append(removed_nodes)
            route_segments_list.append(route_segments)
            zoomed_in_coordinates_list.extend(coordinates_list)

    return removed_nodes_list, route_segments_list, zoomed_in_coordinates_list

    # return [random.randint(0, 100) for _ in range(5)]
    # nums = parse_num_tag(llm_model.chat())

    # return [nums]


def concorde_cpu(current_obj: int, sub_sol: List[int]) -> int:
    time.sleep(random.uniform(1, 2))
    return current_obj + sum(sub_sol)


def _chunk(lst: List[int], n_chunks: int) -> Iterable[List[int]]:
    k, m = divmod(len(lst), n_chunks)
    for i in range(n_chunks):
        start = i * k + min(i, m)
        end = (i + 1) * k + min(i + 1, m)
        yield lst[start:end]


def _safe_copy(proxy, default):
    """Return a plain object copy of a Manager proxy or *default* if broken."""
    try:
        return proxy() if callable(proxy) else proxy.value if hasattr(proxy, 'value') else list(proxy)
    except (BrokenPipeError, EOFError, ConnectionRefusedError, AttributeError):
        logging.getLogger().warning("manager proxy unavailable – returning default")
        return default

# ---------------------------------------------------------------------------
# LLM producer
# ---------------------------------------------------------------------------
def peek_queue(q, traj_lock):
    temp = []
    with traj_lock:
        while not q.empty():
            item = q.get()
            temp.append(item)
        # After peeking, put items back
        for item in temp:
            q.put(item)
    return temp

async def _llm_producer(llm_name, args, tsp_instance, llm_selector, pending_subproblem_queue, global_obj, global_sol, sol_lock, obj_lock, selection_traj, deadline, t0, traj_queue, traj_lock,
                        X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES,
                        backup_selector, solution_plotter):

    _configure_logging()
    log = logging.getLogger()

    # ==== simulate the generation of LLM selection
    llm_parser = LLMTextParser()

    while time.time() < deadline:
        with sol_lock:
            current_route = list(global_sol)
        with obj_lock:
            current_obj = global_obj.value
        # with traj_lock:
        #     prior_selection = peek_queue(traj_queue)

        current_route_edges = tsp_instance.get_route_pair(current_route)
        removed_nodes_list, route_segments_list, coordinates_list = llm_io(args=args,
                                                                                 solution_plotter=solution_plotter,
                                                                                 current_route=current_route,
                                                                                 current_route_edges=current_route_edges,
                                                                                 pending_subproblem_queue=pending_subproblem_queue,
                                                                                 tsp_instance=tsp_instance,
                                                                                 prior_selection = selection_traj,
                                                                                 llm_selector=llm_selector,
                                                                                 backup_selector=backup_selector,
                                                                                 llm_parser=llm_parser,
                                                                                 num_subregion=args.llm_subproblem_selection,
                                                                                 x_min=X_MIN,
                                                                                 x_max=X_MAX,
                                                                                 y_min=Y_MIN,
                                                                                 y_max=Y_MAX,
                                                                                 grid_res=GRID_RES,
                                                                                 num_nodes=0,
                                                                                 traj_lock=traj_lock
                                                                                     )

        try:
            subproblem = Subproblem(removed_nodes_list=removed_nodes_list,
                                    route_segments_list=route_segments_list,
                                    coordinates_list=coordinates_list,
                                    current_obj=current_obj,
                                    solution_version=None,
                                    llm_source=llm_name,
                                    )
            with traj_lock:
                pending_subproblem_queue.put(subproblem)
        except (BrokenPipeError, EOFError):
            log.warning("queue closed – producer exiting early")
            return
        log.info("produced %s from obj=%s sol_len=%s",
                 coordinates_list, current_obj, len(current_route))

    log.info("producer finished (deadline)")

