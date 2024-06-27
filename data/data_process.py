"""
python -m src.data.data_process
"""

from datasets import load_dataset, Dataset
from rich import print
import pandas as pd
import random
import contextlib
from tqdm import tqdm
from typing import List, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
import os
import sys
import json 


def load_data():
    # load dataset from csv file
    dataset = load_dataset("")["train"] # redacted for anonymity purposes
    c13k_problems = pd.read_json("", orient='index') # redacted for anonymity purposes
    c13k_problems = Dataset.from_pandas(c13k_problems)
    return dataset, c13k_problems

def add_problem_id(example):
    example["problem_id"] = problem_id[example["idx"]]
    return example

def problem_to_dict(dataset):
    dataset_dict = {}
    for example in dataset:
        problem_id = example["problem_id"]
        if problem_id not in dataset_dict:
            dataset_dict[problem_id] = []
        dataset_dict[problem_id].append(example)
    return dataset_dict

def main():
    dataset, c13k_problems = load_data()
    print(f"dataset: {dataset}")
    print(f"c13k_problems: {c13k_problems}")
    
    answer = load_dataset('json', data_files='') # redacted for anonymity purposes
    answer = answer['train']
    problem_id = [data['Problem'] for data in dataset]
    answer = answer.map(lambda x, problem_id: {"problem_id": problem_id}, with_indices=True)
        
    with open("", 'w') as f: # redacted for anonymity purposes
        for line in answer:
            f.write(json.dumps(line) + '\n')
    
    dataset_dict = problem_to_dict(answer)

    output_file = "" # redacted for anonymity purposes
    with open(output_file, 'w') as f:
        json.dump(dataset_dict, f)

if __name__ == "__main__":
    main()  
    print(f"sys.argv: {sys.argv}")