import sys
import os
#sys.path.append('/sciclone/home/yli95/codes/vitsp')
#sys.path.append('/workspace/codes/vitsp')
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import multiprocessing as mp
import time
import logging
import psutil
import ctypes
import csv
import asyncio
import numpy as np
from queue import Empty
import pandas as pd
import queue
from dataclasses import asdict, replace
from multiprocessing import Queue
from pathlib import Path
import json

from LLM_TSP.solver.solver import subproblem_solver, subproblem_verifier, sample_independent_subproblem, sample_next_subproblem, GlobalObjRecord   # single line import
from LLM_TSP.config import LLMConfig, SolverConfig
#from LLM_TSP.ablation_config import instance_max_nodes, instance_time_budget

from LLM_TSP.tsp import TravelingSalesmenProblem
from helper.parse_instances import FileParser
from LLM_TSP.initial_solution import initialize_solution
from LLM_TSP.llm import GPT, RoundRobinLLMSelector, MODEL_TYPES
from helper.plot_solution import SolutionPlot
from LLM_TSP.selector import RandomSelector
from LLM_TSP.llm_selector.llm_selector import _llm_producer
from LLM_TSP.llm import LocalInternVL


INTERN_URL = "https://chat.intern-ai.org.cn/api/v1/"

# def _configure_logging() -> None:
#     logging.basicConfig(level=logging.INFO,
#                         format="%(asctime)s [%(processName)s] %(message)s",
#                         datefmt="%H:%M:%S",
#                         force=True)

def _configure_logging(instance_name="tsp_experiment") -> None:
    log_dir = Path("/workspace/codes/vitsp/experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{instance_name}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
        handlers=[
            logging.StreamHandler(),            # 终端继续印
            logging.FileHandler(log_path)       # 同步存入文件
        ]
    )

class PrintLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout  # 记住原本的屏幕输出
        self.log = open(filename, 'w', encoding='utf-8') # 自动创建或覆盖日志文件
    def write(self, message):
        self.terminal.write(message) # 写到屏幕
        self.log.write(message)      # 写到文件
        self.log.flush()             # 实时刷新，防止断电丢失
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
import multiprocessing as mp
import time
import logging
from typing import Dict, List
def _launch_worker(subproblem, config, active_processes):
    proc = mp.Process(
        target=subproblem_solver,
        args=(subproblem, config),
        daemon=True,
        name=f"SubTSP‑{subproblem.id}",
    )
    proc.start()
    active_processes[proc] = subproblem
    return proc

def _reap_finished(active_processes: Dict[mp.Process, "Subproblem"]):
    for proc in list(active_processes.keys()):
        if not proc.is_alive():
            proc.join(timeout=0.1)  # non‑blocking on already‑exited child
            active_processes.pop(proc, None)

def dynamic_worker_manager(config):
    logger = logging.getLogger(__name__)

    capacity_check = lambda: (
            (config.pending_re_subproblem_queue.qsize() > 0 or
            config.pending_ft_subproblem_queue.qsize() > 0) or 
            (config.gain_subproblem_queue.qsize() >0)
            and len(active_processes) < 1
        )
    
    active_processes: Dict[mp.Process, "Subproblem"] = {}
    
    try:
        while time.time() < config.deadline:
            if capacity_check():

                with config.sol_lock, config.obj_lock:
                    current_route =list(config.global_sol)
                    current_obj = config.global_obj.value

                if config.gain_subproblem_queue.qsize() >0:
                    print('Gain subproblem is available!')
                subproblem = sample_independent_subproblem(config=config,
                                            active_subproblems=list(active_processes.values()),
                                            gain_subproblem_queue=config.gain_subproblem_queue,
                                            subproblem_ft_queue=config.pending_ft_subproblem_queue,
                                            subproblem_re_queue=config.pending_re_subproblem_queue,
                                            traj_lock=config.traj_lock,
                                            current_route=current_route
                                            )

                if subproblem is None:
                    time.sleep(1)  # short back‑off when no viable task
                else:
                    subproblem.solution_version = current_obj
                    # _launch_worker(subproblem, config, active_processes)
                    proc = mp.Process(target=subproblem_solver,
                                      args=(subproblem, config),
                                      name=f"SubTSP‑{subproblem.solution_version}",)
                    proc.start()
                    active_processes[proc] = subproblem
                    
                    logger.debug(
                        "Launched %s (alive=%d)",
                        subproblem.solution_version,
                        len(active_processes),
                    )

            _reap_finished(active_processes)
            # print('Active subproblem is ', len(active_processes))
            time.sleep(1)

    finally:
        # ----------------------------------------------------------
        # Final clean‑up: wait for *all* still‑running children
        # ----------------------------------------------------------
        for proc in list(active_processes.keys()):
            proc.join()
        active_processes.clear()
        logger.info("Manager shut‑down complete – final objective = %s", config.global_obj.value)


def verifier_manager(config):
    logger = logging.getLogger(__name__)

    capacity_check = lambda: (
            (config.pending_re_subproblem_queue.qsize() > 0 or
            config.pending_ft_subproblem_queue.qsize() > 0)
            and len(active_processes) < config.args.max_workers
        )
    
    active_processes: Dict[mp.Process, "Subproblem"] = {}
    
    try:
        while time.time() < config.deadline:
            if capacity_check():

                with config.sol_lock, config.obj_lock:
                    current_route =list(config.global_sol)
                    current_obj = config.global_obj.value

                #TODO: modify the samply strategy
                subproblem = sample_next_subproblem(config=config,
                                            active_subproblems=list(active_processes.values()),
                                            subproblem_ft_queue=config.pending_ft_subproblem_queue,
                                            subproblem_re_queue=config.pending_re_subproblem_queue,
                                            traj_lock=config.traj_lock,
                                            current_route=current_route
                                            )

                if subproblem is None:
                    time.sleep(1)  # short back‑off when no viable task
                else:
                    subproblem.solution_version = current_obj
                    # _launch_worker(subproblem, config, active_processes)
                    proc = mp.Process(target=subproblem_verifier,
                                      args=(subproblem, config),
                                      name=f"SubTSP‑{subproblem.solution_version}",)
                    proc.start()
                    active_processes[proc] = subproblem
                    
                    logger.debug(
                        "Launched %s (alive=%d)",
                        subproblem.solution_version,
                        len(active_processes),
                    )

            _reap_finished(active_processes)
            print('Active subproblem is ', len(active_processes))
            time.sleep(1)

    finally:
        for proc in list(active_processes.keys()):
            proc.join()
        active_processes.clear()
        logger.info("Manager shut‑down complete – final objective = %s", config.global_obj.value)
                    
def launch_llm_process(name,config):

    # instance_name = Path(config.args.instance_path).stem
    # _configure_logging(instance_name)
    # log = logging.getLogger()

    instance_name = Path(config.args.instance_path).stem
    log_dir = Path("/workspace/codes/vitsp/experiments/logs")
    log_file_path = log_dir / f"{instance_name}_print.log"
    sys.stdout = PrintLogger(log_file_path)
    sys.stderr = sys.stdout
    
    print(f"[{name}] started llm process")

    asyncio.run(_llm_producer(
        name,
        config.args,
        config.tsp_instance,
        config.llm_selector,
        config.pending_subproblem_queue,
        config.global_obj,
        config.global_sol,
        config.sol_lock,
        config.obj_lock,
        config.selection_traj,
        config.deadline,
        config.t0,
        config.traj_queue,
        config.traj_lock,
        config.X_MIN,
        config.X_MAX,
        config.Y_MIN,
        config.Y_MAX,
        config.GRID_RES,
        config.backup_selector,
        config.solution_plotter
    ))
    #log.info("complete the llm process")

    print(f"[{name}] complete the llm process")


def tsp_instance_initializer(args):

    def determine_instance_boundary(coordinates):
        MARGIN = 0
        x_min = min(coord[0] for coord in coordinates) - MARGIN
        x_max = max(coord[0] for coord in coordinates) + MARGIN
        y_min = min(coord[1] for coord in coordinates) - MARGIN
        y_max = max(coord[1] for coord in coordinates) + MARGIN

        grid_resolution = max((x_max - x_min), (y_max - y_min)) // 10

        return x_min, x_max, y_min, y_max, grid_resolution

    file_parser = FileParser()
    instance_info = file_parser.parse_instance_from_file(args.instance_path)
    coordinates = instance_info['COORDINATES']
    distance_mat = np.array(instance_info['COST_MATRIX'])
    nodes = {i: (x, y) for i, (x, y) in enumerate(coordinates)}
    X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES = determine_instance_boundary(coordinates)
    boundary_info = (X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES)
    tsp_instance = TravelingSalesmenProblem(node_coords_dict=nodes, distance_mat=distance_mat)

    return tsp_instance, boundary_info
# def dump_global_obj_queue(q: Queue, csv_path: str | Path) -> pd.DataFrame:
#     """
#     Drain *q* (containing `GlobalObjRecord`s) into a DataFrame
#     and save it to *csv_path*.

#     Returns
#     -------
#     pd.DataFrame
#         The dataframe that was written, so you can keep using it.
#     """
#     records = []
#     while True:
#         try:
#             rec = q.get_nowait()          # type: GlobalObjRecord
#             records.append(asdict(rec))
#         except queue.Empty:
#             break

#     df = pd.DataFrame(records)
#     df.to_csv(csv_path, index=False)
#     return df

def dump_global_obj_queue(q: Queue, csv_path: str | Path):
    """
    实时追加记录到 CSV 文件中。
    如果文件不存在，则先写入表头。
    """
    records_saved = 0
    file_exists = os.path.isfile(csv_path)
    
    while True:
        try:
            # get_nowait 会非阻塞地从队列拿数据，如果没有就报错 Empty
            rec = q.get_nowait()
            rec_dict = asdict(rec)
            
            with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rec_dict.keys())
                # 只有当文件刚创建，且是第一条记录时才写表头
                if not file_exists and records_saved == 0:
                    writer.writeheader()
                writer.writerow(rec_dict)
                
            records_saved += 1
            file_exists = True # 写完第一条后，文件肯定存在了
            
        except queue.Empty:
            break # 队列空了，退出循环
            
    return records_saved

def pin(proc: mp.Process, cores: list[int]) -> None:
    """Bind *proc* to the given CPU *cores*."""
    psutil.Process(proc.pid).cpu_affinity(cores)

def save_experiment_report(config, routes, coordinates, instance_name):
    """
    整合打印逻辑：生成最终路径图和模型决策热力图
    """
    import pandas as pd
    from pathlib import Path

    save_dir = Path("/workspace/codes/vitsp/experiments/plots")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 图 1：最终路径图 (证明结果) ---
    print(f"Generating final tour plot for {instance_name}...")
    fig_final = config.solution_plotter.plot_tsp_solution(
        routes=routes,
        coordinates=coordinates,
        x_min=config.X_MIN, x_max=config.X_MAX,
        y_min=config.Y_MIN, y_max=config.Y_MAX,
        grid_resolution=config.GRID_RES
    )
    fig_final.savefig(save_dir / f"{instance_name}_final_tour.png", dpi=300, bbox_inches='tight')
    config.solution_plotter.close_fig(fig_final)

    # --- fig 2：决策热力图 (展示过程) ---
    # 从 selection_traj 队列中提取所有 LLM 选框的数据
    records = []
    while not config.selection_traj.empty():
        try:
            records.append(config.selection_traj.get_nowait())
        except:
            break
            
    if records:
        print(f"Generating attention heatmap for {instance_name}...")
        # 将数据转为 DataFrame 以匹配 subrectangles_heatmap 的输入
        # 假设记录里包含 'Subrectangle Trajectory' 字段
        traj_df = pd.DataFrame(records)
        
        fig_heat = config.solution_plotter.subrectangles_heatmap(
            routes=routes,
            coordinates=coordinates,
            spreadheat_data=traj_df,
            x_min=config.X_MIN, x_max=config.X_MAX,
            y_min=config.Y_MIN, y_max=config.Y_MAX,
            grid_resolution=config.GRID_RES
        )
        fig_heat.savefig(save_dir / f"{instance_name}_decision_heatmap.png", dpi=300, bbox_inches='tight')
        config.solution_plotter.close_fig(fig_heat)
    else:
        print("No selection trajectory found, skipping heatmap.")

    print(f"Report saved to {save_dir}")

def main(args):
    instance_name = Path(args.instance_path).stem

    log_dir = Path("/workspace/codes/vitsp/experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"{instance_name}_print.log"

    sys.stdout = PrintLogger(log_file_path)
    sys.stderr = sys.stdout
    #_configure_logging(instance_name)
    #log = logging.getLogger()
    tsp_instance, boundary_info = tsp_instance_initializer(args)
    print('Instance is initialized!')
    X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES = boundary_info 

    #fast_thinking_llm_selector = RoundRobinLLMSelector([GPT(OPENAI_API_1, MODEL_TYPES[args.fast_llm_model], base_url=INTERN_URL)])
    local_model_path = "/workspace/codes/vitsp/InternVL3_5-8B-Flash" 
    fast_thinking_llm_selector = RoundRobinLLMSelector([LocalInternVL(model_path=local_model_path)])
    
    reasoning_llm_selector = RoundRobinLLMSelector([GPT(OPENAI_API_2, MODEL_TYPES[args.reasoning_llm_model], base_url=INTERN_URL)])

    # --- 新增：LLM 连通性冒烟测试 ---
    print("="*50)
    print("[TEST] 正在验证 LLM API 连通性...")
    try:
        # 测试 Fast Thinking LLM (Intern-VL)
        test_selector = fast_thinking_llm_selector.get_next_llm()
        response = test_selector.generate("Please reply 'API OK'.")
        print(f"[SUCCESS] Fast LLM ({args.fast_llm_model}) 响应: {response}")

        # 测试 Reasoning LLM (Intern-Reasoning)
        test_reasoning = reasoning_llm_selector.get_next_llm()
        response_re = test_reasoning.generate("Please reply 'Reasoning OK'.")
        print(f"[SUCCESS] Reasoning LLM ({args.reasoning_llm_model}) 响应: {response_re}")
    except Exception as e:
        print(f"[FATAL] LLM 调用失败! 错误信息: {e}")
        # sys.exit(1) 
    print("="*50)
    # --- 测试结束 ---

    backup_selector = RandomSelector(model_name='random')
    solution_plotter = SolutionPlot()

    instance_name = Path(args.instance_path).stem
     # Solution initialization
    try:
        with open(f'/workspace/codes/vitsp/experiments/LKH_solutions/{instance_name}_solution.json', 'r') as f:
            data = json.load(f)

            current_route = data['current_route']
            current_obj = data['current_obj']
            warmstart_latency = data['warmstart_latency']
    except:
        current_route, current_obj, warmstart_latency = initialize_solution(args, tsp_instance)
    # -------- solution initialization (warm start)

    # Save to file
        data = {
            'current_route': current_route,
            'current_obj': current_obj,
            'warmstart_latency': warmstart_latency
        }
        

        with open(f'/workspace/codes/vitsp/experiments/LKH_solutions/{instance_name}_solution.json', 'w') as f:
            json.dump(data, f)
    # -------- shared queues and values among parallel processes
    gain_subproblem_queue       = mp.Queue() # used to save subproblems with definite gains
    pending_ft_subproblem_queue = mp.Queue() # save the pending subproblems from fast thinking LLM
    pending_re_subproblem_queue = mp.Queue() # from reasoning LLM
    track_global_obj_queue      = mp.Queue() # save the whole trajectory to track the global solution improvement
    selection_traj              = mp.Queue()
    global_obj                  = mp.Value('i', 0) # creating using manager.Value may cause broken pipe
    global_sol                  = mp.Array(ctypes.c_int, len(current_route), lock=True) # to avoid broken pipe when concorde did not succeed in finding optimal tour
    obj_lock                    = mp.RLock() # use the lock may not be a good idea
    sol_lock                    = mp.RLock()
    traj_lock                   = mp.RLock()
    solver_proc_lock            = mp.RLock()
    
    with obj_lock:
        global_obj.value = current_obj
    with sol_lock:
        global_sol[:] = current_route[:]

    now = round(warmstart_latency, 2)
    record = GlobalObjRecord(latency=now,
                             new_obj=current_obj,
                             coords=None,
                             num_nodes_removed=None,
                             llm_mode=args.initial_solution_model,
                             global_solution_version=None,
                             )
    track_global_obj_queue.put(record)
            

    t0 = time.time()
    deadline = t0 + args.total_time_budget - warmstart_latency 

    solver_config = SolverConfig(args=args,
                                 warmstart_latency=warmstart_latency,
                                 tsp_instance=tsp_instance,
                                 pending_ft_subproblem_queue=pending_ft_subproblem_queue,
                                 pending_re_subproblem_queue=pending_re_subproblem_queue,
                                 gain_subproblem_queue = gain_subproblem_queue,
                                 global_obj=global_obj,
                                 global_sol=global_sol,
                                 obj_lock=obj_lock,
                                 sol_lock=sol_lock,
                                 traj_lock=traj_lock,
                                 solver_proc_lock=solver_proc_lock,
                                 selection_traj=selection_traj,
                                 t0=t0,
                                 deadline=deadline,
                                 track_global_obj_queue=track_global_obj_queue
                                 )
    ft_llm_config = LLMConfig(args=args,
                           tsp_instance=tsp_instance,
                           llm_selector=fast_thinking_llm_selector,
                           pending_subproblem_queue=pending_ft_subproblem_queue,
                           global_obj=global_obj,
                           global_sol=global_sol,
                           sol_lock=sol_lock,
                           obj_lock=obj_lock,
                           selection_traj=selection_traj,
                           deadline=deadline,
                           t0=t0,
                           traj_queue=track_global_obj_queue,
                           traj_lock=traj_lock,
                           X_MIN=X_MIN,
                           X_MAX=X_MAX,
                           Y_MIN=Y_MIN,
                           Y_MAX=Y_MAX,
                           GRID_RES=GRID_RES,
                           backup_selector=backup_selector,
                           solution_plotter=solution_plotter
                           )
    
    re_llm_config = replace(ft_llm_config,
                            llm_selector=reasoning_llm_selector,
                            pending_subproblem_queue=pending_re_subproblem_queue,)

    
    #TODO: simulating putting some subproblems in the queue
    
    dynamic_solver_proc      = mp.Process(name='Concorde', 
                                     target=dynamic_worker_manager,
                                     args=(solver_config,))

    verifier_proc            = mp.Process(name='Concorde', 
                                     target=verifier_manager,
                                     args=(solver_config,))
    
    reasoning_llm_proc       = mp.Process(name="reasoning_LLM-Producer", 
                                     target=launch_llm_process,
                                     args=('reasoning', re_llm_config,))
    
    fast_thinking_llm_proc   = mp.Process(name="fast_thinking_LLM-Producer", 
                                     target=launch_llm_process,
                                     args=('fast_thinking', ft_llm_config,))
    
    
    n_cpus = mp.cpu_count()  # e.g. 48
    llm1_core = [0, 1]  # one core for each LLM producer
    llm2_core = [2, 3]
    # solver_cores = list(range(4, n_cpus))  # the rest for Concorde


    reasoning_llm_proc.start()
    fast_thinking_llm_proc.start()
    dynamic_solver_proc.start()

    
    verifier_proc.start()


    # pin(dynamic_solver_proc, solver_cores)
    pin(fast_thinking_llm_proc, llm1_core)
    pin(reasoning_llm_proc, llm2_core)

    # dynamic_solver_proc.join()
    dynamic_solver_proc.join(timeout=args.total_time_budget + 10)
    if dynamic_solver_proc.is_alive():
        print("dynamic_solver_proc is stuck!")
        dynamic_solver_proc.terminate()
        dynamic_solver_proc.join()

    instance_name = Path(args.instance_path).stem

    saved_count = dump_global_obj_queue(track_global_obj_queue,
                            f'/workspace/codes/vitsp/experiments/LLM_TSP_exp/{instance_name}_max_nodes_{args.max_node_for_solver}_time_budget_{args.total_time_budget}_initial_{args.initial_solution_model}_llm_{args.fast_llm_model}_{args.reasoning_llm_model}_solver_{args.solver_model}_subproblem_{args.llm_subproblem_selection}_parallel_workers.csv')
    print("Saved", saved_count, "records")

    
    verifier_proc.join(timeout=5)
    if verifier_proc.is_alive():
        print("verifier_proc is stuck!")
        verifier_proc.terminate()
        verifier_proc.join()

    
    reasoning_llm_proc.join(timeout=10)
    if reasoning_llm_proc.is_alive():
        print("reasoning_llm_proc is stuck!")
        reasoning_llm_proc.terminate()
        reasoning_llm_proc.join()

    fast_thinking_llm_proc.join(timeout=5)
    if fast_thinking_llm_proc.is_alive():
        print("fast_thinking_llm_proc is stuck!")
        fast_thinking_llm_proc.terminate()
        fast_thinking_llm_proc.join()

    print("All processes finished. Generating visual reports...")
    # 注意：使用 list(global_sol) 确保数据被正确读取
    save_experiment_report(
        config=solver_config, 
        routes=list(global_sol), 
        coordinates=tsp_instance.node_coords, 
        instance_name=instance_name
    )

    # instance_name = Path(args.instance_path).stem

    # df = dump_global_obj_queue(track_global_obj_queue,
    #                         f'/local/scratch/a/XXXX-1/vllm-carbon-XXXX-5/LLM-TSP-async/experiments/LLM_TSP_exp/{instance_name}_max_nodes_{args.max_node_for_solver}_time_budget_{args.total_time_budget}_initial_{args.initial_solution_model}_llm_{args.fast_llm_model}_{args.reasoning_llm_model}_solver_{args.solver_model}_subproblem_{args.llm_subproblem_selection}_parallel_workers.csv')
    # print("Saved", len(df), "records")

    # fast_llm_proc.join()
    # dynamic_solver_proc.join()
    
    # # reasoning_llm_proc.join()
    # print('complete')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Traveling Salesmen Problem Solver")
    parser.add_argument('--instance_path', type=str,
                        default='/workspace/codes/vitsp/instances/tsplib/tsplib_repo',
                        help='Path to the instance file')
    parser.add_argument('--max_iterations', type=int, default=5,
                        help='Maximum number of iterations for optimization')
    parser.add_argument('--total_time_budget', type=float, default=2000,
                        help='Wall time in seconds')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of solvers working in parallel')
    # ---------------------------------------------------------------------------
    # Initializer Specification
    # ---------------------------------------------------------------------------
    parser.add_argument('--initial_solution_model', type=str, default='LKH',
                        help='model to generate initial solution')

    # ---------------------------------------------------------------------------
    # Solver Specification
    # ---------------------------------------------------------------------------
    parser.add_argument('--solver_model', type=str, default='concorde',
                        help='solver name for reoptimization')
    parser.add_argument('--SolverTimeLimit', type=float, default=10,
                        help='Time allowed for Concorde solver')
    parser.add_argument('--max_node_for_solver', type=int, default=1000,
                        help='Max number of nodes sent to solver')

    # ---------------------------------------------------------------------------
    # Selector Specification
    # ---------------------------------------------------------------------------
    parser.add_argument('--fast_llm_model', type=str, default='intern-vl', 
                        help='LLM model name for selector, qwen2.5-32b-v, gpt-4o, gpt-4.1-2025-04-14 ') 
    parser.add_argument('--reasoning_llm_model', type=str, default='intern-reasoning', 
                        help='LLM model name for selector, qwen2.5-32b-v, gpt-4o, o4-mini-2025-04-16 ') 
    parser.add_argument('--keep_selection_trajectory', action='store_true',
                        help='whether incorporating selection trajectory for llm')
    parser.add_argument('--llm_subproblem_selection', type=int, default=2,
                        help='number of subproblems that LLM should select in its first try')
    parser.add_argument('--select_sequence', action='store_true',
                        help='whether selecting a sequence or rectangle as subproblem')
    parser.add_argument('--random_selection', action='store_true',
                        help='whether selecting a sequence or rectangle as subproblem')
    parser.add_argument('--hard_coded_subrectangle', action='store_true',
                        help='Flag to enable hard-coded subrectangle')
    parser.add_argument('--gridding_resolution', type=int, default=5,
                        help='divide the plot into K if wanting to fix')

    args = parser.parse_args()

    file_path = args.instance_path
    print('The instance path is ', file_path)
    tsp_files = [
        # 'dsj1000.tsp',
        #'pr1002.tsp',
        # 'u1060.tsp',
        # 'vm1084.tsp',
        # 'pcb1173.tsp',
        # 'd1291.tsp',
        # 'rl1304.tsp',
        # 'rl1323.tsp',
        # 'nrw1379.tsp',
        # 'fl1400.tsp',
        # 'u1432.tsp',
        # 'fl1577.tsp',
        # 'd1655.tsp',
        # 'vm1748.tsp',
        # 'u1817.tsp',
        # 'rl1889.tsp',
        # 'd2103.tsp',
        # 'u2152.tsp',
        # 'u2319.tsp',
        # 'pr2392.tsp',
        # 'pcb3038.tsp',
        # 'fl3795.tsp',
        # 'fnl4461.tsp',
        # 'rl5915.tsp',
        # 'rl5934.tsp',
        # 'pla7397.tsp',
        # 'rl11849.tsp',
        # 'usa13509.tsp',
        # 'brd14051.tsp',
        # 'd15112.tsp',
        # 'd18512.tsp',
        'pla33810.tsp',
        # 'pla85900.tsp',
    ]

    for file in tsp_files:
        args.instance_path = f'{file_path}/{file}'
        print(f"Processing instance: {args.instance_path}")
        main(args)
