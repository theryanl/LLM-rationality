import os
import sys
import ast
import json 
import argparse
from tqdm import tqdm
import torch
from pathlib import Path
from random import random
from dataclasses import dataclass
from typing import Literal, Optional, Union, Tuple
from src.data.data_utils import *
from rich import print
import time
from vllm import LLM, SamplingParams

def get_in_context(sample, label="A"):
    input_text = sample.input_text
    input_text += label
    return input_text


@torch.inference_mode()
def inference(args):
    model_name = args.model_name    
    llm = LLM(model=model_name, 
              tokenizer=model_name,
              tensor_parallel_size=4)
    
    
    # get the prompts
    dataset = ChoicesDataset(
        num_samples=args.num_examples,
        seed=args.seed,
    )
    samples = dataset.samples
    # print(f"samples: {samples}")
    in_context_samples = ""
    label_A = get_in_context(samples[-1], label="A")
    label_B = get_in_context(samples[-1], label="B")
    in_context_samples = label_A + "\n" + label_B + "\n"
    if "instruct" in model_name:
        samples = [s for s in samples if "instruct" in s.input_text]
    else:
        input_prompts = [in_context_samples + s.input_text for s in samples]
        
    sampling_params = SamplingParams(max_tokens=10, 
                                     temperature=0.2,
                                     top_p = 0.9,
                                     ignore_eos=False)
    start = time.perf_counter()
    outputs = llm.generate(input_prompts, sampling_params=sampling_params)
    tot_time = time.perf_counter() - start
    print(f"tot_time in minutes: {tot_time / 60}")
    for out in outputs:
        generated_text = out.outputs[0].text
        print(f"generated_text: {generated_text}")
        print(f"output: {out}")
        input()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="data/choices13k")
    parser.add_argument("--model_name", type=str, default="") # redacted for anonymity purposes
    args = parser.parse_args()
    inference(args)
    print(f"[bold green]Inference completed![/bold green]")