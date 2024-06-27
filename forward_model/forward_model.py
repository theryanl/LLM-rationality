"""
python -m src.forward_model.forward_model
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
from src.forward_model.claude_utils import *
from src.data.data_utils import *
from rich import print


def main(args):
    # get the prompts
    dataset = ChoicesDataset(
        begin_idx = args.begin_idx,
        num_samples=args.num_examples,
        seed=args.seed,
        instruction_mode=args.instruction_mode,
    )
    samples = dataset.samples
    # print(f"samples: {samples}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{args.begin_idx}_{args.begin_idx + args.num_examples}_choices13k.jsonl"
    if os.path.exists(output_file):
        check = input(f"File {output_file} already exists. Overwrite? (y/n): ")
        if check.lower() != "y":
            sys.exit()
    
    if args.prompt_method == "one_completion":
        with open(output_file, "a") as f:
            for sample in tqdm(samples, desc="forward modelling"):
                prompt = [sample.input_text]
                # agent = OpenAI_Run(
                #     temperature=args.temperature,
                #     max_tokens=args.max_tokens,
                # )
                agent = Claude3()
                
                print(f"agent: {agent}")
                agent.system_prompt = args.system_prompt if args.system_prompt else None    
                completion = agent.complete(prompt)
                print(f"completion: {completion}")
                # input()
                answer = []
                print(f"completion: {completion}")
                
                # For act as individual cot
                if ('"choice": "A"' in completion or '"Choice": "A"' in completion) and ('"choice": "B"' not in completion and '"Choice": "B"' not in completion):
                    answer.append("A")
                elif ('"choice": "B"' in completion or '"Choice": "B"' in completion) and ('"choice": "A"' not in completion and '"Choice": "A"' not in completion):
                    answer.append("B")
                else:
                    answer.append(None)
               
                b_rate = sample.b_rate
                a_rate = 1 - b_rate
                
                output = dict(
                    sample_id=sample.sample_id,
                    prompt=sample.input_text,
                    completion=completion,
                    answer=answer,
                    b_rate=b_rate,
                    a_rate=a_rate,
                )
                print(f"output: {output}")
                f.write(json.dumps(output) + "\n")
    elif args.prompt_method == 'cot':
        with open(output_file, "a") as f:
            for sample in tqdm(samples, desc="forward modelling"):
                answer = []
                cot = []
                for n in tqdm(range(sample.num_p), desc="forward modelling by n samples"):
                    prompt = [sample.input_text]
                    agent = OpenAI(
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    agent.system_prompt = args.system_prompt if args.system_prompt else None    
                    completion = agent.complete(prompt)
                    print(f"completion: {completion}")
                    cot.append(completion)
                    
                    # For act as individual cot
                    if ('"choice": "A"' in completion or '"Choice": "A"' in completion) and ('"choice": "B"' not in completion and '"Choice": "B"' not in completion):
                        answer.append("A")
                    elif ('"choice": "B"' in completion or '"Choice": "B"' in completion) and ('"choice": "A"' not in completion and '"Choice": "A"' not in completion):
                        answer.append("B")
                    else:
                        answer.append(None)
                    print(f"answer: {answer}")
                
                num_A = answer.count("A") / sample.num_p
                num_B = answer.count("B") / sample.num_p
                std_B = np.std([1 if a == "B" else 0 for a in answer])
                std_A = np.std([1 if a == "A" else 0 for a in answer])
                num_none = answer.count(None) / sample.num_p
                prob = dict(
                    Prob_A=num_A,
                    Prob_B=num_B,
                    std_A=std_A,
                    std_B=std_B,
                    num_none=num_none,
                )
                b_rate = sample.b_rate
                a_rate = 1 - b_rate
                b_rate_std = sample.b_rate_std
                output = dict(
                    prompt=sample.input_text,
                    completion=completion,
                    answer=answer,
                    prob=prob,
                    num_p=sample.num_p,
                    b_rate=b_rate,
                    a_rate=a_rate,
                    b_rate_std=b_rate_std,
                    cot=cot if cot else None,
                )
                print(f"output: {output}")
                f.write(json.dumps(output) + "\n")
        print(f"Data written to {output_file}")
        
    elif args.prompt_method == 'zero_shot':
        with open(output_file, "a") as f:
            for sample in tqdm(samples, desc="forward modelling"):
                answer = []
                cot = []
                completions = []
                for n in tqdm(range(sample.num_p), desc="forward modelling by n samples"):
                    prompt = [sample.input_text]
                    agent = OpenAI(
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    agent.system_prompt = args.system_prompt if args.system_prompt else None    
                    completion = agent.complete(prompt)
                    if "A" in completion:
                        answer.append("A")
                    elif "B" in completion:
                        answer.append("B")
                    else:
                        answer.append(None)
                    completions.append(completion)
                
                num_A = answer.count("A") / sample.num_p
                num_B = answer.count("B") / sample.num_p
                std_B = np.std([1 if a == "B" else 0 for a in answer])
                std_A = np.std([1 if a == "A" else 0 for a in answer])
                num_none = answer.count(None) / sample.num_p
                prob = dict(
                    Prob_A=num_A,
                    Prob_B=num_B,
                    std_A=std_A,
                    std_B=std_B,
                    num_none=num_none,
                )
                b_rate = sample.b_rate
                a_rate = 1 - b_rate
                b_rate_std = sample.b_rate_std
                output = dict(
                    prompt=sample.input_text,
                    completion=completions,
                    answer=answer,
                    prob=prob,
                    num_p=sample.num_p,
                    b_rate=b_rate,
                    a_rate=a_rate,
                    b_rate_std=b_rate_std,
                    cot=cot if cot else None,
                )
                print(f"output: {output}")
                f.write(json.dumps(output) + "\n")
        print(f"Data written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin_idx", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=14568)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="data/choices13k")
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--prompt_method", type=str, default="cot")
    parser.add_argument("--instruction_mode", type=str, default="individual_zero_shot")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1)
    args = parser.parse_args()
    main(args)
    print(f"[bold green] Completed forward modelling [/bold green]")
    