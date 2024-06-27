import os
os.environ ['CUDA_LAUNCH_BLOCKING'] ='1'
import json
import yaml
from pathlib import Path
import time
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
from rich import print
from datetime import datetime
import argparse
from datasets import load_dataset, load_from_disk

import argparse
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data.data_utils import *
import transformers
from rich import print
import time


def setup_model(model_name, max_seq_length):
    config = transformers.AutoConfig.from_pretrained(
        model_name,
    )

    config.use_cache = False

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_seq_length=max_seq_length,
        padding_side='right',
        trust_remote_code=True,
    )
        
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


@torch.inference_mode()
def inference(args):
    model, tokenizer = setup_model(args.model_name, max_seq_length=args.max_seq_length)
    
    if tokenizer.eos_token_id is not None:
        model.config.pad_token_id = tokenizer.eos_token_id
    model.to('cuda')
    tokenizer.pad_token = tokenizer.eos_token
    # get the prompts
    dataset = ChoicesDataset(
        num_samples=args.num_examples,
        seed=args.seed,
    )
    samples = dataset.samples

    input_prompts = [s.input_text for s in samples]
    pipeline = transformers.pipeline('text-generation', model=args.model_name, model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "choices13k.jsonl"
    if os.path.exists(output_file):
        check = input(f"File {output_file} already exists. Overwrite? (y/n): ")
        if check.lower() != "y":
            sys.exit()
            
    if 'instruct' in args.model_name.lower():
        with open(output_file, "a") as f:
            for prompt, sample in tqdm(zip(input_prompts, samples), desc="inference", total = len(samples)):
                num_iter = sample.num_p
                answer = []
                for i in tqdm(range(num_iter)):
                    messages = [
                        {"role": "system", "content": "Please select one Mahcine from the following option. Only return A or B as the answer:"},
                        {"role": "user", "content": prompt},
                    ]
                    prompt = pipeline.tokenizer.apply_chat_template(messages,
                                                                    tokenize=False,
                                                                    add_generation_prompt=True)
                    terminators = [
                            pipeline.tokenizer.eos_token_id,
                            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        ]
                    outputs = pipeline(
                        prompt,
                        max_new_tokens=2,
                        eos_token_id=terminators,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                    # print(f"outputs: {outputs}")
                    out = outputs[0]["generated_text"][len(prompt):]
                    print(f"output: {out}")
                    answer.append(outputs[0]["generated_text"][len(prompt):])
                
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
                    answer=answer,
                    prob=prob,
                    num_p=sample.num_p,
                    b_rate=b_rate,
                    a_rate=a_rate,
                    b_rate_std=b_rate_std,
                )
                f.write(json.dumps(output) + "\n")
    else:
        for prompt, sample in tqdm(zip(input_prompts, samples), desc="inference"):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
            attention_mask = (input_ids != tokenizer.pad_token_id).float().to(input_ids.device)
        
            input_length = input_ids.shape[1]
            
            generated_tokens = model.generate(input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=1,
                                num_return_sequences=1)
            generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            print(f"generated_text: {generated_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="data/choices13k_llama3_instruct_5000")
    parser.add_argument("--model_name", type=str, default="") # redacted for anonymity purposes
    parser.add_argument("--max_seq_length", type=int, default=1024)
    args = parser.parse_args()
    inference(args)
    print(f"[bold green]Inference completed![/bold green]")