"""
python -m src.forward_model.forward_model_batch
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
from src.data.data_utils import *
from rich import print
from openai import OpenAI 


def make_batch_item(sample, max_tokens, temperature = 0.0):
    prompt = sample.input_text
    batch_item = dict(
        custom_id=str(sample.sample_id),
        method="POST",
        url="/v1/chat/completions",
        body=dict(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=80,
            temperature=temperature,
        ),
    )
    return batch_item



def get_batch_items(args, samples, output_file):
    with open(output_file, "w") as f:
        for sample in tqdm(samples, desc="forward modelling"):
            batch_item = make_batch_item(sample, args.max_tokens, args.temperature)
            f.write(json.dumps(batch_item) + "\n")
    print(f"Data written to {output_file}")
    
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
    output_path = Path(args.output_dir,args.instruction_mode)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{args.begin_idx}_{args.begin_idx + args.num_examples}_choices13k.jsonl"
    
    # save to the batch file
    get_batch_items(args, samples, output_file)
    
    client = OpenAI(
        api_key="", # redacted for anonymity purposes
    )
    batch_input_file = client.files.create(
        file=open(output_file, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    batch_object = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "forward modeling"
        }
    )
    print(f"Batch job created: {batch_object}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin_idx", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=14568)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="") # redacted for anonymity purposes
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--prompt_method", type=str, default="zero_shot")
    parser.add_argument("--instruction_mode", type=str, default="aggregate_zero_shot")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=80)
    args = parser.parse_args()
    main(args)
    client = OpenAI(
        api_key="", # redacted for anonymity purposes
    )
    
    batch_id = "" # redacted for anonymity purposes
    batch_status = client.batches.retrieve(batch_id)
    print(f"Batch job status: {batch_status}")
    
    # cancel the batch job
    client.batches.cancel('') # redacted for anonymity purposes
# 
    print(f"[bold green] Completed forward modelling [/bold green]")
    