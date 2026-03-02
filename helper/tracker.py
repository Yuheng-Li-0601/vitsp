#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created on 10/30/24 8:34 PM
@File:tracker.py
@Author:XXXX-6
@Contact: XXXX-1@XXXX-7.edu
'''

import time
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = '/local/scratch/a/XXXX-1/vllm-carbon-XXXX-5/LLM-TSP-async/experiments/LLM_TSP_exp'

# OUTPUT_DIR
class MetricsTracker:
    def __init__(self, solver_name, selector_name):
        self.solver_name = solver_name
        self.selector_name = selector_name
        self.global_start_time = None
        self.iteration_start_time = None
        self.global_time = 0
        self.iteration_latencies = []
        self.selector_latencies = []
        self.solver_latencies = []
        self.objective_values = []
        self.subrectangle_trajectory = []
        self.num_removed_nodes = []

    def save_plot(self, plot):
        # save the plots as the input to LLM
        pass

    def start_global_timer(self):
        self.global_start_time = time.time()

    def stop_global_timer(self):
        if self.global_start_time is not None:
            self.global_time = time.time() - self.global_start_time


    def start_iteration_timer(self):
        self.iteration_start_time = time.time()

    def stop_iteration_timer(self):
        if self.iteration_start_time is not None:
            iteration_time = time.time() - self.iteration_start_time
            self.iteration_latencies.append(iteration_time)

    def log_selector_latency(self, latency):
        self.selector_latencies.append(latency)
    def add_selector_latency(self, latency):
        self.selector_latencies[-1] += latency

    def log_solver_latency(self, latency):
        self.solver_latencies.append(latency)

    def log_objective_value(self, value):
        self.objective_values.append(value)

    def log_subrectangle(self, subrectangle):
        self.subrectangle_trajectory.append(subrectangle)

    def log_iteration_latency(self, latency):
        self.iteration_latencies.append(latency)
    def log_num_removed_nodes(self, num_removed_nodes):
        self.num_removed_nodes.append(num_removed_nodes)

    def save_iteration_plot(self, tsp_plot, iteration, plot_dir='plots'):
        plot_path = f"../../{plot_dir}/tsp_plot_iteration_{iteration}.png"
        tsp_plot.savefig(plot_path)
        plt.close(tsp_plot)

    # def export_to_csv(self, csv_path='metrics.csv'):
    #     with open(csv_path, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(['Metric', 'Values'])
    #         writer.writerow(['Global Time', self.global_time])
    #         writer.writerow(['Iteration Times', self.iteration_latencies])
    #         writer.writerow(['Selector Latencies', self.selector_latencies])
    #         writer.writerow(['Solver Latencies', self.solver_latencies])
    #         writer.writerow(['Objective Values', self.objective_values])
    #         writer.writerow(['Subrectangle Trajectory', self.subrectangle_trajectory])

    def export_to_csv(self, args):
        instance_name = Path(args.instance_path).stem

        trajectory_status = 'keep_selection_traj' if args.keep_selection_trajectory else 'no_selection_traj'

        if args.select_sequence:
            csv_path = (
                    OUTPUT_DIR
                    + f'/{instance_name}_iter_{args.max_iterations}_{args.initial_solution_model}_'
                    + f'random_sequence_max_n{args.max_node_for_solver}_{args.solver_model}_co_subsequence{args.llm_subproblem_selection}_solvertimelimit_{args.TimeLimit}_grid{args.gridding_resolution}'
                    + f'.csv'
            )
        else:
            csv_path = (
                    OUTPUT_DIR
                    + f'/{instance_name}_iter_{args.max_iterations}_{args.initial_solution_model}_'
                    + f'{args.llm_model}_{args.solver_model}_co_subregion{args.llm_subproblem_selection}_'
                    + f'{trajectory_status}_max_n{args.max_node_for_solver}_solvertimelimit_{args.TimeLimit}_grid{args.gridding_resolution}.csv'
            )

        data = {
            'Iteration Latencies': self.iteration_latencies,
            'Selector Latencies': self.selector_latencies,
            'Solver Latencies': self.solver_latencies,
            'Objective Values': self.objective_values,
            'Subrectangle Trajectory': self.subrectangle_trajectory,
            'Num Removed Nodes': self.num_removed_nodes
        }

        df = pd.DataFrame(data)
        try:
            # Attempt to save the DataFrame to CSV
            df.to_csv(csv_path, index=False)
        except:
            try:
                df.to_csv(csv_path)
            except AttributeError as e:
                print(f"Error occurred: {e}")

                # Handle the specific error related to '_format_native_types'
                if "'Index' object has no attribute '_format_native_types'" in str(e):
                    print("Fixing column index issues...")

                    # Convert column names to strings
                    df.columns = df.columns.map(str)

                    # Reset index if necessary
                    if isinstance(df.columns, pd.MultiIndex):
                        print("Detected MultiIndex in columns. Resetting index...")
                        df = df.reset_index()

                    # Retry saving the DataFrame
                    try:
                        df.to_csv(csv_path, index=False)
                        print("File saved successfully after fixing column index issues.")
                    except Exception as retry_exception:
                        print(f"Retry failed: {retry_exception}")
                else:
                    # Re-raise the exception if it's not the expected one
                    np.save(OUTPUT_DIR + f'/{instance_name}_iter_{args.max_iterations}_{args.initial_solution_model}_{args.llm_model}_{args.solver_model}.npy', df)

    def export_final_image(self, args, fig):
        instance_name = Path(args.instance_path).stem
        fig.write_image(f'{OUTPUT_DIR}/{instance_name}_iter_{args.max_iterations}_{args.initial_solution_model}_{args.llm_model}_{args.solver_model}_tsp_{Path(args.instance_path).stem}.png', engine='kaleido')

    def report_metrics(self):
        print(f"Global Time: {self.global_time:.2f} s")
        print(
            f"Average Iteration Time: {sum(self.iteration_latencies) / len(self.iteration_latencies):.2f} s" if self.iteration_latencies else "No iterations recorded.")
        print(f"Final Objective Value: {self.objective_values[-1] if self.objective_values else 'N/A'}")
