#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created on 10/26/24 5:59 PM
@File:parse_llm_response.py
@Author:XXXX-6
@Contact: XXXX-1@XXXX-7.edu
'''

import re
import pandas as pd

class LLMTextParser:
    def clean_trace_content_from_llm(self, text):
        # Find all contents within <trace>...</trace>
        matches = re.findall(r'<trace>(.*?)</trace>', text, re.DOTALL)
        cleaned_lists = []

        for match in matches: # only parse the first matched trace
            # Remove unwanted characters and split into a list of numbers
            clean_list = re.findall(r'\d+', match)
            # Convert each string number to an integer
            clean_list = list(map(int, clean_list))
            cleaned_lists.append(clean_list)

        # Return cleaned lists if found, otherwise None
        return cleaned_lists[0] if cleaned_lists else None

    @staticmethod
    def parse_subrectangle_coordinates(text):
        # able to handle <coordinate> mixed with other content
        coordinate_pattern = re.compile(
            r"<coordinates>\s*"
            r"x_min\s*=\s*([\d,]+)\s*,\s*"
            r"x_max\s*=\s*([\d,]+)\s*,\s*"
            r"y_min\s*=\s*([\d,]+)\s*,\s*"
            r"y_max\s*=\s*([\d,]+)\s*"
            r"</coordinates>",
            re.DOTALL | re.IGNORECASE
        )

        # Find all matches for the coordinate pattern in the text
        matches = coordinate_pattern.findall(text)
        coordinates_list = []

        # Convert each match (tuple of strings) into integers, handling commas in numbers
        for match in matches:
            x_min, x_max, y_min, y_max = [int(value.replace(",", "")) for value in match]
            coordinates_list.append((x_min, x_max, y_min, y_max))

        # If no matches were found, return a list with a single tuple of None values
        if not coordinates_list:
            coordinates_list.append((None, None, None, None))

        return coordinates_list

if __name__ == "__main__":
    parse = LLMTextParser()
    data = pd.read_csv('/Users/XXXX-4/XXXX-14/XXXX-13/XXXX-10-XXXX-7.edu/XXXX-11/XXXX-16/Meta-XXXX-15-3-8B-Instruct_batch_1_n_20.csv')
    data['Cleaned Trace'] = data['Output Response'].apply(parse.clean_trace_content_from_llm)
