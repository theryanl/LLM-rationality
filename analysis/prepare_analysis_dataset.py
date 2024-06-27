
"""
python -m analysis.prepare_analysis_dataset
"""
import os
import sys
import ast
import json 
import argparse
from tqdm import tqdm
from pathlib import Path
from random import random
from dataclasses import dataclass
from typing import Literal, Optional, Union, Tuple
from typing_extensions import Annotated
from src.forward_model.openai_utils import *
from src.data.data_utils import *
from rich import print

def cal_expected_value(prob_A, prob_B):
    """
    Calculate the expected value of A and B
    Parameters:
        prob_A: list [[prob, reward], [prob, reward], ...]
        prob_B: list [[prob, reward], [prob, reward], ...]
    """
    expected_A = 0
    expected_B = 0
    for prob, reward in prob_A:
        expected_A += round(prob, 4) * reward
    for prob, reward in prob_B:
        expected_B += round(prob, 4) * reward
    return expected_A, expected_B


def main(args):
    model_dataset_path = "" # redacted for anonymity purposes
    model_dataset = load_dataset('json', data_files=model_dataset_path)['train']
    print(f"model_dataset: {model_dataset}")
    dataset = ChoicesDataset(
        begin_idx = args.begin_idx,
        num_samples=args.num_examples,
        seed=args.seed,
        instruction_mode=args.instruction_mode,
    )
    human_samples = dataset.samples
    samples = dataset.reward_samples
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    output_file = output_dir / f"choices13k.jsonl"
    with open(output_file, "w") as f:
        if args.instruction_mode == "individual_cot":
            count = 0
            for sample, human_sample, model_pred in zip(samples, human_samples, model_dataset):
                expected_A, expected_B = cal_expected_value(sample.prob_A, sample.prob_B)
                completion = model_pred["completion"]
                prob_A = 1 if model_pred['answer'][0] == "A" else 0
                prob_B = 1 if model_pred['answer'][0] == "B" else 0
                prob = {"Prob_A": prob_A, "Prob_B": prob_B}
                output = dict(
                    sample_id= count,
                    prompt=model_pred['prompt'],
                    completion=model_pred['completion'],
                    answer=model_pred['answer'],
                    prob=[prob],
                    b_rate=model_pred['b_rate'],
                    a_rate=model_pred['a_rate'],
                )
                f.write(json.dumps(output) + "\n")
                count += 1
                
        elif args.instruction_mode == "individual_zero_shot":
            for sample, human_sample, model_pred in zip(samples, human_samples, model_dataset):
                completion = model_pred["completion"]
                prob_A = 1 if model_pred['answer'][0] == "A" else 0
                prob_B = 1 if model_pred['answer'][0] == "B" else 0
                prob = {"Prob_A": prob_A, "Prob_B": prob_B}
                output = dict(
                    sample_id=model_pred['sample_id'],
                    prompt=model_pred['prompt'],
                    completion=model_pred['completion'],
                    answer=model_pred['answer'],
                    prob=[prob],
                    b_rate=model_pred['b_rate'],
                    a_rate=model_pred['a_rate'],
                )
                f.write(json.dumps(output) + "\n")
                
        elif args.instruction_mode == "act_individual_cot":
            for sample, human_sample, model_pred in zip(samples, human_samples, model_dataset):
                prob_A = 1 if model_pred['answer'][0] == "A" else 0
                prob_B = 1 if model_pred['answer'][0] == "B" else 0
                prob = {"Prob_A": prob_A, "Prob_B": prob_B}
                output = dict(
                    sample_id=model_pred['sample_id'],
                    prompt=model_pred['prompt'],
                    completion=model_pred['completion'],
                    answer=model_pred['answer'],
                    prob=[prob],
                    b_rate=model_pred['b_rate'],
                    a_rate=model_pred['a_rate'],
                )
                f.write(json.dumps(output) + "\n")
                
        elif args.instruction_mode == "act_individual_zero_shot":
            for sample, human_sample, model_pred in zip(samples, human_samples, model_dataset):
                expected_A, expected_B = cal_expected_value(sample.prob_A, sample.prob_B)
                completion = model_pred["completion"]
                prob_A = 1 if model_pred['answer'][0] == "A" else 0
                prob_B = 1 if model_pred['answer'][0] == "B" else 0
                prob = {"Prob_A": prob_A, "Prob_B": prob_B}
                output = dict(
                    sample_id=model_pred['sample_id'],
                    prompt=model_pred['prompt'],
                    completion=model_pred['completion'],
                    answer=model_pred['answer'],
                    prob=[prob],
                    b_rate=model_pred['b_rate'],
                    a_rate=model_pred['a_rate'],
                )
                f.write(json.dumps(output) + "\n")

        elif args.instruction_mode == "aggregate_zero_shot" or args.instruction_mode == "aggregate_cot":
            count = 0
            for sample, human_sample, model_pred in zip(samples, human_samples, model_dataset):
                expected_A, expected_B = cal_expected_value(sample.prob_A, sample.prob_B)
                completion = model_pred["completion"][0]
                try:
                    begin_idx = completion.find('{\n')
                    end_idx = completion.find('\n}')
                    final_completion = completion[begin_idx:end_idx+len('\n}')]
                    final_completion = json.loads(final_completion)
                    machine_A_prob = final_completion["Machine A"]
                    machine_B_prob = final_completion["Machine B"]
                    if type(machine_A_prob) == int and type(machine_B_prob) == int:
                        model_choices = "A" if machine_A_prob > machine_B_prob else "B"
                        machine_A_prob = machine_A_prob / 100
                        machine_B_prob = machine_B_prob / 100
                        
                    if type(machine_A_prob) == float and type(machine_B_prob) == float:
                        model_choices = "A" if machine_A_prob > machine_B_prob else "B"
                        machine_A_prob = machine_A_prob
                        machine_B_prob = machine_B_prob
                    elif "%" in machine_A_prob and "%" in machine_B_prob:
                        machine_A_prob = int(machine_A_prob.replace("%", ""))
                        machine_B_prob = int(machine_B_prob.replace("%", ""))
                        model_choices = "A" if machine_A_prob > machine_B_prob else "B"
                    elif "%" in machine_A_prob and "%" not in machine_B_prob:
                        machine_A_prob = int(machine_A_prob.replace("%", ""))
                        model_choices = "A" if machine_A_prob > machine_B_prob else "B"
                    elif "%" not in machine_A_prob and "%" in machine_B_prob:
                        machine_B_prob = int(machine_B_prob.replace("%", ""))
                        model_choices = "A" if machine_A_prob > machine_B_prob else "B"
                    elif type(machine_A_prob) == str and type(machine_B_prob) == str:
                        machine_A_prob = int(machine_A_prob.replace("%", ""))
                        machine_B_prob = int(machine_B_prob.replace("%", ""))
                        model_choices = "A" if machine_A_prob > machine_B_prob else "B"
                    
                    else:
                        model_choices = None
                    prob= {"Prob_A": machine_A_prob / 100, "Prob_B": machine_B_prob / 100}
                except:
                    if model_pred['completion'] is None:
                        print(f"Model pred: {model_pred}")
                        model_choices=None
                        prob={"Prob_A": 0, "Prob_B": 0}
                    elif 'A": 100' in model_pred['completion'] or 'A": 95' in model_pred['completion'] \
                        or 'A": 90' in model_pred['completion'] or 'A": 85' in model_pred['completion'] \
                        or 'A": 80' in model_pred['completion'] or 'A": 75' in model_pred['completion'] \
                        or 'A": 70' in model_pred['completion'] or 'A": 65' in model_pred['completion'] \
                        or 'A": 60' in model_pred['completion'] or 'A": 55' in model_pred['completion'] \
                        or 'A": "Majority' in model_pred['completion'] or 'A": "majority' in model_pred['completion'] \
                        or 'A": "Higher Percentage' in model_pred['completion'] or 'A": "higher percentage' in model_pred['completion'] \
                        or 'a higher preference for Machine A' in model_pred['completion'] or 'more people might be inclined to choose Machine A' in model_pred['completion'] \
                        or 'majority would choose Machine A' in model_pred['completion'] or 'majority would prefer Machine A' in model_pred['completion'] \
                        or 'more might choose Machine A' in model_pred['completion'] \
                        or 'would choose the guaranteed payout of Machine A' in model_pred['completion']:
                        model_choices="A"
                        prob={"Prob_A": 1, "Prob_B": 0}
                    elif '"Machine B": 100' in model_pred['completion'] or '"machine B": 100' in model_pred['completion'] \
                        or 'B": 95' in model_pred['completion'] or 'B": 90' in model_pred['completion'] \
                        or 'B": 85' in model_pred['completion'] or 'B": 80' in model_pred['completion'] \
                        or 'B": 75' in model_pred['completion'] or 'B": 70' in model_pred['completion'] \
                        or 'B": 65' in model_pred['completion'] or 'B": 60' in model_pred['completion'] \
                        or 'B": 55' in model_pred['completion'] or 'B": 50' in model_pred['completion'] \
                        or 'B": "Majority' in model_pred['completion'] or 'B": "majority' in model_pred['completion'] \
                        or 'B": "Higher Percentage' in model_pred['completion'] or 'B": "higher percentage' in model_pred['completion'] \
                        or 'more people might be inclined to choose Machine B' in model_pred['completion'] or 'a higher preference for Machine B' in model_pred['completion'] \
                        or 'majority would choose Machine B' in model_pred['completion'] or 'majority would prefer Machine B' in model_pred['completion'] \
                        or 'more might choose Machine B' in model_pred['completion'] \
                        or 'would choose the guaranteed payout of Machine B' in model_pred['completion']:
                        model_choices="B"
                        prob={"Prob_A": 0, "Prob_B": 1}
                  
                output = dict(
                    sample_id=count,
                    prompt=model_pred['prompt'],
                    completion=model_pred['completion'],
                    answer=[model_choices],
                    prob=[prob],
                    b_rate=model_pred['b_rate'],
                    a_rate=model_pred['a_rate'],
                )

                f.write(json.dumps(output) + "\n")
                count += 1
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin_idx", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=14568)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="") # redacted for anonymity purposes
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--prompt_method", type=str, default="cot")
    parser.add_argument("--instruction_mode", type=str, default="aggregate_cot")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1)
    args = parser.parse_args()
    main(args)
    print(f"[bold green] Completed forward modelling analysis preparation [/bold green]")
    