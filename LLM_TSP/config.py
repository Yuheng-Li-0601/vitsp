from dataclasses import dataclass
import asyncio

@dataclass
class LLMConfig:
    args: any
    tsp_instance: any
    llm_selector: any
    pending_subproblem_queue: any
    global_obj: any
    global_sol: any
    sol_lock: any
    obj_lock: any
    selection_traj: any
    deadline: float
    t0: float
    traj_queue: any
    traj_lock: any
    X_MIN: float
    X_MAX: float
    Y_MIN: float
    Y_MAX: float
    GRID_RES: int
    backup_selector: any
    solution_plotter: any


@dataclass
class SolverConfig:
    args: any
    warmstart_latency: float
    tsp_instance: any
    pending_ft_subproblem_queue: any
    pending_re_subproblem_queue: any
    gain_subproblem_queue: any
    global_obj: any
    global_sol: any
    obj_lock: any
    sol_lock: any
    traj_lock: any
    solver_proc_lock: any
    selection_traj: any
    deadline: float
    t0: float
    track_global_obj_queue: any
    