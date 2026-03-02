#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created on 11/3/24 4:39 PM
@File:selector.py
@Author:XXXX-6
@Contact: XXXX-1@XXXX-7.edu
'''

import random
import numpy as np

class RandomSelector():
    def __init__(self, model_name="random"):
        self.model_name = model_name

    @staticmethod
    def generate_subrectangle(X_MIN=0, X_MAX=1000, Y_MIN=0, Y_MAX=1000, no_more_than=200, num_region=2):
        # Generate random coordinates within the range [0, 1000]
        coordinates_list = []
        for _ in range(num_region):
            X_MIN, X_MAX, Y_MIN, Y_MAX = int(X_MIN), int(X_MAX), int(Y_MIN), int(Y_MAX)
            x_min = random.randint(X_MIN, X_MAX)  # Ensure x_max can be larger
            x_max = random.randint(x_min, X_MAX)  # x_max must be greater than x_min
            y_min = random.randint(Y_MIN, Y_MAX)  # Ensure y_max can be larger
            y_max = random.randint(y_min, Y_MAX)  # y_max must be greater than y_min

            # Format the result as required
            coordinates = f"<coordinates> x_min= {x_min}, x_max= {x_max}, y_min= {y_min}, y_max= {y_max} </coordinates>"
            coordinates_list.append((x_min, x_max, y_min, y_max))
        return coordinates_list

    def generate_random_subrectangle_with_fixed_size(self):
        # the size of the subrectangle is proportional to the whole instance
        pass

    @staticmethod
    def generate_sequence(route, max_length=200, randomize=False, head_weights=None, length_weights=None):
        """
        Generates a cohesive sequence of nodes from a route based on a normal distribution or weighted selection.

        Parameters:
        - route (list): The list of route nodes.
        - max_length (int): The maximum length of the sequence.
        - head_weights (list or None): Optional weights for selecting the head node index.
        - length_weights (list or None): Optional weights for selecting the sequence length.

        Returns:
        - list: A randomized subsequence from the route.
        """
        if not route:
            raise ValueError("Route cannot be empty")

        route_length = len(route)

        # Select the head node index.
        if head_weights:
            if len(head_weights) != route_length:
                raise ValueError("Length of head_weights must match length of route")
            head_index = random.choices(range(route_length), weights=head_weights, k=1)[0]
        else:
            head_index = random.randint(0, route_length - 1)

        # Shuffle the route while keeping head node at the start.
        shuffled_route = route[head_index:] + route[:head_index]

        # Determine the sequence length.
        if length_weights:
            possible_lengths = list(range(1, min(max_length, route_length) + 1))
            if len(length_weights) != len(possible_lengths):
                raise ValueError("Length of length_weights must match possible sequence lengths")
            seq_length = random.choices(possible_lengths, weights=length_weights, k=1)[0]
        else:
            if randomize:
                mean_length = max_length // 2
                std_dev = max_length // 3
                seq_length = int(np.clip(np.random.normal(mean_length, std_dev), 1, max_length))
            else:
                seq_length = max_length

        # Clip the sequence length to avoid going beyond the route length.
        if seq_length > len(shuffled_route):
            seq_length = len(shuffled_route)

        # Return the final sequence.
        return shuffled_route[:seq_length]