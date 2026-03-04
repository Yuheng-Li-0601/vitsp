#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created on 9/2/24 12:04 PM
@File:gurobi_model.py
@Author:Zhuoli Yin
@Contact: yin195@purdue.edu
'''
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from itertools import combinations, combinations_with_replacement
from typing import List, Tuple, Dict


def cvrp_model(num_customers, demands, vehicle_capacity, cost_matrix, num_vehicles):
    # Set of customers (excluding depot)
    customers = list(range(1, num_customers + 1))
    # Set of nodes including depot (0 and num_ori_nodes+1 representing the depot)
    nodes = [0] + customers

    # Create the model
    model = gp.Model("CVRP")

    # ================ Decision variables
    x = model.addVars(nodes, nodes, vtype=GRB.BINARY, name="origin_dist_mat")
    y = model.addVars(nodes, vtype=GRB.CONTINUOUS, lb=0, name="y")

    # =============== Objective function: minimize the total cost
    model.setObjective(
        gp.quicksum(cost_matrix[i][j] * x[i, j] for i in nodes for j in nodes),
        GRB.MINIMIZE
    )

    # ============ Constraints
    # Each customer must be visited exactly once, return to a node and depart from this node
    # Flow conservation constraints
    model.addConstrs(
        (gp.quicksum(x[i, j] for j in nodes if j != i) == 1 for i in customers),
        name="visit_once_depart"
    )
    model.addConstrs(
        (gp.quicksum(x[i, j] for i in nodes if j != i) == 1 for j in customers),
        name="visit_once_return"
    )

    # Vehicle limit leaving the depot
    model.addConstr(
        gp.quicksum(x[0, j] for j in customers) == num_vehicles,
        name="vehicle_limit_depart"
    )
    model.addConstr(
        gp.quicksum(x[j, 0] for j in customers) == num_vehicles,
        name="vehicle_limit_return"
    )

    # Subtour elimination constraints (to ensure valid routes)
    model.addConstrs(
        ((x[i, j] == 1) >> (y[i] + demands[j - 1] == y[j])
         for i in customers for j in customers),
        name="subtour_elimination"
    )

    # Capacity constraints
    model.addConstrs(
        (demands[i - 1] <= y[i] for i in customers),
        name="capacity_ub"
    )
    model.addConstrs(
        (y[i] <= vehicle_capacity for i in customers),
        name="capacity_lb"
    )

    return model, x, y, nodes


def tsp_model(nodes: List, distance_mat: np.ndarray,
              fixed_routes: Dict[Tuple[int, int], int] = None):
    model = gp.Model("TSP")
    x = {}  # defining x_{i,j} as a binary variable
    n = len(nodes)

    if fixed_routes is None:
        fixed_routes = {}

    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = model.addVar(
                    obj=distance_mat[i, j], vtype=GRB.BINARY, name=f'x_{i}_{j}'
                )
                # Fix variable if it's in fixed_routes
                if (i, j) in fixed_routes:
                    x[i, j].ub = 1
                    x[i, j].lb = 1

    # Each city must be arrived at from exactly one other city
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(n) if j != i) == 1,
            'arrive' + str(i)
        )

    # Each city must be departed to exactly one other city
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[j, i] for j in range(n) if j != i) == 1,
            'depart' + str(i)
        )

    # Subtour elimination constraints (using lazy constraints)
    def subtour_elimination(model, where):
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._vars)
            selected = gp.tuplelist(
                (i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5
            )
            tour = get_subtour(selected)
            if len(tour) < n:
                model.cbLazy(
                    gp.quicksum(
                        model._vars[i, j] + model._vars[j, i]
                        for i, j in combinations(tour, 2)
                    ) <= len(tour) - 1
                )

    # Given a tuplelist of edges, find the shortest subtour
    def get_subtour(edges):
        unvisited = nodes[:]
        cycle = nodes[:]  # Dummy - guaranteed to be replaced
        while unvisited:
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*')
                             if j in unvisited]
            if len(thiscycle) <= len(cycle):
                cycle = thiscycle
        return cycle

    model.setObjective(
        gp.quicksum(distance_mat[i, j] * x[i, j] for i, j in x.keys()),
        GRB.MINIMIZE
    )
    model._vars = x
    model.Params.lazyConstraints = 1
    model.optimize(subtour_elimination)

    return model, x


def get_tsp_solution_route(x, n):
    """Retrieve the TSP route from the optimized solution."""
    selected_edges = [(i, j) for i, j in x.keys() if x[i, j].x > 0.5]
    route = []
    visited = set()
    current_node = 0

    while len(route) < n:
        route.append(current_node)
        visited.add(current_node)
        try:
            next_node = next(
                j for i, j in selected_edges
                if i == current_node and j not in visited
            )
        except StopIteration:
            break
        current_node = next_node

    route.append(route[0])
    return route


def solve_cvrp(model, x, y, nodes):
    model.optimize()
    if model.status == GRB.OPTIMAL:
        solution = model.getAttr('origin_dist_mat', x)
        routes = []
        for i in nodes:
            for j in nodes:
                if solution[i, j] > 0.5:
                    routes.append((i, j))
        return routes, model.ObjVal
    else:
        return None, None


def check_infeasibility(model):
    """
    Check the feasibility of a Gurobi model and analyze infeasibility if detected.

    Args:
        model (gurobipy.Model): The Gurobi model to check.

    Returns:
        None
    """
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible. Performing infeasibility analysis...")
        model.computeIIS()
        model.write("infeasibility_report.ilp")
        print("Infeasibility report written to 'infeasibility_report.ilp'")
        print("The following constraints are part of the IIS:")
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"Constraint {c.constrName} is infeasible.")
        for v in model.getVars():
            if v.IISLB > 0 or v.IISUB > 0:
                print(f"Variable {v.varName} is part of the IIS.")
    elif model.status == GRB.OPTIMAL:
        print("Model is feasible. Optimal solution found.")
    else:
        print(f"Model status: {model.status}")


class GurobiTSPModel:
    def __init__(self, nodes: List, distance_mat: np.ndarray):
        self.nodes = nodes
        self.distance_mat = distance_mat
        self.model = gp.Model("TSP")
        self.x = {}
        self.fixed_routes = {}
        self._tsp_model()
        self.model.update()

    def _tsp_model(self):
        n = len(self.nodes)
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.x[i, j] = self.model.addVar(
                        obj=self.distance_mat[i, j],
                        vtype=GRB.BINARY,
                        name=f'x_{i}_{j}'
                    )

        # Each city must be arrived at from exactly one other city
        for i in range(n):
            self.model.addConstr(
                gp.quicksum(self.x[i, j] for j in range(n) if j != i) == 1,
                'arrive' + str(i)
            )

        # Each city must be departed to exactly one other city
        for i in range(n):
            self.model.addConstr(
                gp.quicksum(self.x[j, i] for j in range(n) if j != i) == 1,
                'depart' + str(i)
            )

        self.model.setObjective(
            gp.quicksum(
                self.distance_mat[i, j] * self.x[i, j]
                for i, j in self.x.keys()
            ),
            GRB.MINIMIZE
        )
        self.model._vars = self.x
        self.model.Params.lazyConstraints = 1

    def update_model_param(self, args):
        """Pass args to update model parameters."""
        self.model.setParam('MIPFocus', args.MIPFocus)
        self.model.setParam('Presolve', args.Presolve)
        self.model.setParam('Cuts', args.Cuts)
        self.model.setParam('Heuristics', args.gurobi_heuristics)
        self.model.setParam('TimeLimit', args.TimeLimit)

    def subtour_elimination(self, model, where):
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._vars)
            selected = gp.tuplelist(
                (i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5
            )
            tour = self.get_subtour(selected)
            if len(tour) < len(self.nodes):
                model.cbLazy(
                    gp.quicksum(
                        model._vars[i, j] + model._vars[j, i]
                        for i, j in combinations(tour, 2)
                    ) <= len(tour) - 1
                )

    def get_subtour(self, edges):
        unvisited = self.nodes[:]
        cycle = self.nodes[:]
        while unvisited:
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [
                    j for i, j in edges.select(current, '*') if j in unvisited
                ]
            if len(thiscycle) < len(cycle):
                cycle = thiscycle
        return cycle

    def optimize(self):
        self.model.optimize(self.subtour_elimination)

    def update_fixed_routes(self, new_fixed_routes: Dict[Tuple[int, int], int]):
        """Update bounds based on changes between old and new fixed routes."""
        old_fixed_routes = self.fixed_routes

        # Unfix the variables that are no longer in the new fixed routes
        for (i, j) in old_fixed_routes.keys() - new_fixed_routes.keys():
            if (i, j) in self.x:
                self.x[i, j].lb = 0
                self.x[i, j].ub = 1

        # Fix the new routes that were not fixed previously
        for (i, j) in new_fixed_routes.keys() - old_fixed_routes.keys():
            if (i, j) in self.x:
                self.x[i, j].lb = 1
                self.x[i, j].ub = 1

        self.fixed_routes = new_fixed_routes
        self.model.update()

    def get_tsp_route(self):
        """Retrieve the TSP route from the optimized solution."""
        n = len(self.nodes)
        selected_edges = [
            (i, j) for i, j in self.x.keys() if self.x[i, j].x > 0.5
        ]

        route = []
        visited = set()
        current_node = 0

        while len(route) < n:
            route.append(current_node)
            visited.add(current_node)
            try:
                next_node = next(
                    j for i, j in selected_edges
                    if i == current_node and j not in visited
                )
            except StopIteration:
                break
            current_node = next_node

        route.append(route[0])
        return route

    def get_objective_value(self):
        return self.model.getObjective().getValue()
