# ViTSP: A Vision Language Models Guided Framework for Large-Scale Traveling Salesman Problems

## ViTSP
Use `LLM_TSP/llm_tsp_async.py` as the main code for ViTSP.
The ViTSP needs an OpenAI API key, Concorde, and LKH-3.

## Concorde

`./exact_concorde/exact_concorde.py`

## LKH-3:
For LKH-3 (Default), we use the default parameter values: MAX_TRIALS=instance_dim, RUNS=10. Run `./heuristic_LKH/heuristic_LKH`

For LKH-3 (more RUNS), to increase the value of RUNS and obtain objective values over runtime, run: `./heuristic_LKH/LKH_param_sweeping.py`
# Dataset
TSPLIB instance: `./instances`

It can also be downloaded at [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/).