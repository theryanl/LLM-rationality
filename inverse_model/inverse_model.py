"""
python3 -m src.inverse_model.inverse_model
"""
import argparse
from typing import Literal, Optional, Union, Tuple
from typing_extensions import Annotated
import src.inverse_model.prompts
from src.inverse_model.utils import *
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm
import os
from src.inverse_model.fix_queries import *
from openai import OpenAI
from src.inverse_model.batch import *

def pairwise_experiment(args, client):
    start_time = time.time()
    # get choices. each example contains the bag. The chosen bag is always first, and the target is always "x".
    if args.prompt_type == "positive":
        options_path = "src/inverse_model/positive_weak_to_strong.txt"
    else:
        options_path = "src/inverse_model/negative_weak_to_strong_fixed.txt" # replacing weak_to_strong.txt
    choices = get_choices(args.num_choices, options_path) # this is also the ordering of weakest to strongest for the stimuli. 

    # create pairs of choices
    pairs = []
    for i in range(args.num_choices):
        for j in range(i+1, args.num_choices):
            pair = dict()
            pair["choices"] = [choices[i], choices[j]]
            pair["indices"] = (i, j)
            pairs.append(pair)
    
    ranking_matrix = np.zeros((args.total_completions, args.num_choices, args.num_choices))
    
    # dims: total_completions (shuffles), first choice in pair, second choice in pair
    save = [[dict() for i in range(args.num_choices)] for j in range(args.num_choices)]
    total_cost = 0
    max_tokens = 50
    if args.prompt_method == "cot": max_tokens = 500

    # number of shuffles
    for c_index in range(args.c_index_start, args.c_index_start + args.num_completions):
        print(f"Completion {c_index}")

        # add context (e.g. colors of candy / location of shocks) to each pair, assigned randomly
        for pair in pairs:
            pair = assign_context(pair, args.prompt_type)

        # shuffle ordering of choices in each pair, options in each choice, items in each option
        for pair in pairs:
            pair = shuffle_context(pair)

        # convert pairs to prompts
        prompts = []
        for pair in pairs:
            indices, prompt_text = construct_prompt(pair, args.prompt_type, args.prompt_method)
            prompts.append([indices, prompt_text, pair])

        # prompt the model
        for prompt_index in range(len(prompts)):
            if prompt_index % 10 == 0:
                print(f"{prompt_index}/{len(prompts)}")
            prompt = prompts[prompt_index]
            indices, prompt_text, pair = prompt[0], prompt[1], prompt[2]

            result = None
            while result is None:
                completion = make_chat_call(client, args.model, prompt_text, max_tokens)
                result, response_text, cost = parse_response(completion, args.model, args.prompt_method, pair['target_item'], client)
                total_cost += cost

            if pair["shuffled_ordering"] == [1, 0]:
                result = 1 - result

            ranking_matrix[c_index, indices[0], indices[1]] += result
            ranking_matrix[c_index, indices[1], indices[0]] += 1 - result
            
            save[indices[0]][indices[1]] = {"indices": indices, 
                                            "completion_index": c_index, 
                                            "prompt": prompt_text, 
                                            "response": response_text, 
                                            "model": args.model,
                                            "parsed": result, 
                                            "shuffled_ordering": pair["shuffled_ordering"], 
                                            "target_item": pair["target_item"], 
                                            "choices": pair["choices"], 
                                            "choices_context": pair["choices_context"], 
                                            "shuffled_choices_context_1": pair["shuffled_choices_context"][0].shuffled_choice,
                                            "shuffled_choices_context_2": pair["shuffled_choices_context"][1].shuffled_choice,
                                            "prompt_type": args.prompt_type,
                                            "prompt_method": args.prompt_method, 
                                            }
    
        # save the results
        # get the time for the filename
        now = datetime.now()
        date_string = now.strftime("%m-%d-%H-%M-%S_")
        if not os.path.exists(args.output_dir + str(c_index)+ "/"):
            os.makedirs(args.output_dir + str(c_index) + "/")
        
        output_path = args.output_dir + str(c_index) + "/" + date_string
        output_file = output_path + "ranking_matrix.npy"
        np.save(output_file, ranking_matrix)
        print(f"Ranking matrix saved to {output_file}")

        output_file = output_path + "save.json"
        with open(output_file, "w") as f:
            json.dump(save, f)
        print(f"Results saved to {output_file}")
        print(f"Total cost: {total_cost}")
        print(f"Completed completion {c_index}, {time.time() - start_time} seconds")

def ranking_experiment(args, client):
    # compare all 47 choices at once
    # get choices. each example contains the bag. The chosen bag is always first, and the target is always "x".
    if args.prompt_type == "positive":
        options_path = "src/inverse_model/positive_weak_to_strong.txt"
    else:
        options_path = "src/inverse_model/negative_weak_to_strong_fixed.txt"
    choices = get_choices(args.num_choices, options_path) # this is also the ordering of weakest to strongest for the stimuli.
    uncontextualized_choices = dict()
    uncontextualized_choices["choices"] = choices

    save = [dict() for i in range(args.total_completions)]
    total_cost = 0
    max_tokens = 50
    if args.prompt_method == "cot": max_tokens = 500
    ranking_matrix = np.zeros((args.total_completions, args.num_choices, args.num_choices))

    for c_index in range(args.c_index_start, args.c_index_start + args.total_completions):
        contextualized_choices = assign_context(uncontextualized_choices, args.prompt_type)
        contextualized_choices = shuffle_context(contextualized_choices)
        prompt_text = construct_ranking_prompt(contextualized_choices, args.prompt_type, args.prompt_method)
        completion = make_chat_call(client, args.model, prompt_text, max_tokens) # this is the same
        print(completion.choices[0].message.content)
        raise NotImplementedError
        result, response_text, cost = parse_response(completion) # not implemented
        # result should be a list of 47 indices, from weakest to strongest evidence. 
        total_cost += cost

        # map the results back to before shuffling (this should use unconextualized_choices["shuffled_ordering"])
        
        # fill out the ranking matrix based on the rank results
        # not implemented

        # save the results
        save[c_index] = {"completion_index": c_index, 
                         "prompt": prompt_text, 
                         "response": response_text, 
                         "parsed": result, 
                         "mapping": contextualized_choices["shuffled_ordering"]}
    
    # save the results
    # get the time for the filename
    # ...
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_choices", type=int, default=47)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="src/inverse_model/results/")
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--prompt_type", default="positive")
    parser.add_argument("--prompt_method", default="zero_shot")
    parser.add_argument("--model", type=str, default="gpt-4-0125-preview")
    parser.add_argument("--num_completions", type=int, default=5)
    parser.add_argument("--experiment_type", type=str, default="pairwise", choices=["pairwise", "ranking"])
    parser.add_argument("--c_index_start", type=int, default=0)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--fixing", type=str, default="False", choices=["True", "False"])
    parser.add_argument("--call_method", type=str, default="chat", choices=["chat", "batch"])
    parser.add_argument("--batch_purpose", type=str, choices=["create", "continue", "send", "check", "check_all", "cancel", "results"])
    args = parser.parse_args()

    if args.fixing == "True": args.fixing = True
    else: args.fixing = False

    if args.prompt_type == "positive": args.total_completions = 43
    elif args.prompt_type == "negative": args.total_completions = 42

    return args


if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)

    if args.model == "gpt-4-0125-preview":
        api_key_location=os.path.expanduser(os.path.join("~/.ssh/", f"openai-azure-gpt4-0125-{args.split}"))
        api_base_location=os.path.expanduser(os.path.join("~/.ssh/", f"openai-azure-base-gpt4-0125-{args.split}"))
        client = AzureOpenAI(api_key = open(api_key_location).read().strip(),  
                                api_version = "2023-05-15",
                                azure_endpoint = open(api_base_location).read().strip())
    
    elif args.model == "claude-3-opus-20240229":
        api_key_path = '~/.ssh/anthropic-key-rationality'
        with open(os.path.expanduser(api_key_path), 'r') as f:
            api_key = f.read().strip()
        client = anthropic.Anthropic(api_key=api_key)

    elif args.model == "gpt-4o-2024-05-13":
        api_key_path = '~/.ssh/gpt-rationality'
        with open(os.path.expanduser(api_key_path), 'r') as f:
            api_key = f.read().strip()
        client = OpenAI(api_key=api_key)


    if args.fixing:  
        requery_fixes = [
            {"index": 21, "incorrect": "x/ba/dc", "correct": "x/ba"}, 
            {"index": 25, "incorrect": "cbad/cbax", "correct": "cbax/cbad"},
            {"index": 26, "incorrect": "cbax/cbad", "correct": "x/ba/dc"},
            {"index": 28, "incorrect": "bax/bac/bad", "correct": "bax/c"},
            {"index": 42, "incorrect": "x/a/dcb", "correct": "x/a/b"},
        ]
        requery_results(args, client, requery_fixes) # what needs to be in the "requery_fix"? 
        # index (0-46), previous string (to confirm), new string

    elif args.experiment_type == "pairwise":
        if args.call_method == "chat":
            pairwise_experiment(args, client)

        elif args.call_method == "batch":
            if args.batch_purpose == "create":
                batch_path = pairwise_experiment_batch(args, client)
            
            elif args.batch_purpose == "continue":
                # args only needs output_dir, num_choices, model, batch_purpose, call_method

                # positive zero-shot:
                result_string = "" # redacted for anonymity purposes
                
                batch_continue(args, args.output_dir, result_string)
            

            elif args.batch_purpose == "results":
                # args only needs output_dir, num_choices, model, batch_purpose, call_method

                
                result_string = "" # redacted for anonymity purposes
                previous_updated = "save_requeried" # queried once, gpt-4o parsing

                batch_results(args, args.output_dir, result_string, previous_updated, client)
                
            
            if args.batch_purpose == "send":
                # args only needs model, batch_purpose, call_method

                
                batch_path = "" # redacted for anonymity purposes
                
                send_batch(client, batch_path)

            elif args.batch_purpose == "check":
                
                batch_id = "" # redacted for anonymity purposes
                check_batch(client, batch_id)
            
            elif args.batch_purpose == "check_all":
                batch_list = client.batches.list(limit=10)

                for batch in batch_list:
                    print(batch, "\n\n\n")

            elif args.batch_purpose == "cancel":
                batch_id = ""  # redacted for anonymity purposes
                print(client.batches.cancel(batch_id))

    elif args.experiment_type == "ranking":
        ranking_experiment(args, client)
