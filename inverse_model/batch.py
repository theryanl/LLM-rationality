from src.inverse_model.utils import *
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm
import os
from openai import OpenAI

def make_batch_item(args, c_index, indices, prompt_text, max_tokens):
    batch_item = {
        "custom_id": f"{c_index}_{indices[0]}_{indices[1]}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": args.model,
            "messages": [{'role': 'user', 'content': prompt_text}],
            "max_tokens": max_tokens,
            "temperature": 1,
        }
    }
    return batch_item


def pairwise_experiment_batch(args, client):
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
    
    # dims: total_completions (shuffles), first choice in pair, second choice in pair
    save = [[dict() for i in range(args.num_choices)] for j in range(args.num_choices)]
    max_tokens = 50
    if args.prompt_method == "cot": max_tokens = 500

    batch = []

    # number of shuffles
    for c_index in range(args.c_index_start, args.c_index_start + args.num_completions):
        print(f"Batching completion {c_index}")

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
            prompt = prompts[prompt_index]
            indices, prompt_text, pair = prompt[0], prompt[1], prompt[2]

            batch.append(make_batch_item(args, c_index, indices, prompt_text, max_tokens))
            
            save[indices[0]][indices[1]] = {"indices": indices, 
                                            "completion_index": c_index, 
                                            "prompt": prompt_text, 
                                            "response": None,
                                            "model": args.model,
                                            "parsed": None,
                                            "shuffled_ordering": pair["shuffled_ordering"], 
                                            "target_item": pair["target_item"], 
                                            "choices": pair["choices"], 
                                            "choices_context": pair["choices_context"], 
                                            "shuffled_choices_context_1": pair["shuffled_choices_context"][0].shuffled_choice,
                                            "shuffled_choices_context_2": pair["shuffled_choices_context"][1].shuffled_choice,
                                            "prompt_type": args.prompt_type,
                                            "prompt_method": args.prompt_method, 
                                            }
    
        # get the time for the filename
        now = datetime.now()
        date_string = now.strftime("%m-%d-%H-%M-%S_")
        if not os.path.exists(args.output_dir + str(c_index)+ "/"):
            os.makedirs(args.output_dir + str(c_index) + "/")
        
        output_path = args.output_dir + str(c_index) + "/" + date_string
        output_file = output_path + "save.json"
        with open(output_file, "w") as f:
            json.dump(save, f)
        print(f"save dict saved to {output_file}")
    
    # save the batch
    now = datetime.now()
    date_string = now.strftime("%m-%d-%H-%M-%S_")
    output_file = args.output_dir + date_string + "batch.jsonl"
    with open(output_file, "w") as f:
        for item in batch:
            f.write(json.dumps(item) + "\n")
    print(f"Batch saved to {output_file}, {time.time() - start_time} seconds")


def batch_continue(args, batch_path, result_string):

    # create structs to support finding what was run
    results = []
    found_results = np.zeros((args.total_completions, args.num_choices, args.num_choices))
    # for i in range(args.num_choices):
    #     for j in range(i+1, args.num_choices):
    #         found_results[:, i, j] = 1
    # 0 -> not found, 1 -> found

    # load the output jsonl
    with open(batch_path + result_string, "r") as f:
        lines = f.readlines()
    
    # parse the custom_id field
    for line in lines:
        result = json.loads(line)
        indices = result["custom_id"].split("_")
        c_index = int(indices[0])
        i = int(indices[1])
        j = int(indices[2])
        if i > j: raise ValueError("i > j")

        found_results[c_index, i, j] = 1
        results.append(result)
    
    # construct the remaining queries
    batch = []

    for c_index in range(args.total_completions):

        # load the save_dict to get the prompt
        save_path = batch_path + str(c_index) + "/"
        save_file = None
        for file in os.listdir(save_path):
            if file.endswith(".json") and "05-17-11-48" in file:
                if save_file is not None: raise ValueError("Multiple files in save path, please specify. ")
                save_file = file
        with open(save_path + save_file, "r") as f:
            save_dict = json.load(f)
        
        for i in range(args.num_choices):
            for j in range(i+1, args.num_choices):
                if found_results[c_index, i, j] == 0:

                    # get the item from the save_dict
                    item = save_dict[i][j]
                    prompt_text = item["prompt"]
                    if (item["indices"][0] != i) or (item["indices"][1] != j): raise ValueError(f"Indices do not match, {item['indices']} != {(i, j)}")
                    max_tokens = 50 if item["prompt_method"] == "zero_shot" else 500
                    indices = (i, j)
                    batch.append(make_batch_item(args, c_index, indices, prompt_text, max_tokens))

    print(f"Remaining completions batched: {len(batch)}")
    
    # save the batch
    now = datetime.now()
    date_string = now.strftime("%m-%d-%H-%M-%S_")
    output_file = batch_path + date_string + "batch.jsonl"
    with open(output_file, "w") as f:
        for item in batch:
            f.write(json.dumps(item) + "\n")
    print(f"Batch saved to {output_file}")


def parse_batch_response(response, model, prompt_method, target_item, client):
# Input:
#   response: response from make_chat_call
#   model: string, "gpt-4-..." or "claude-..."
#   prompt_method: string, "cot" or "zero_shot"
#   target_item: string, the context assigned to the target item
#   client: OpenAI / Anthropic client
#
# Returns:
#   result: float in [0, 1]
#   string_response: string, the response from the LLM
#   cost: float, cost in USD
    string_response = response
    string_response_lower = string_response.lower()

    unparsed = 0
    re_queried = False
    if "Choice 1 more strongly suggests" in string_response or \
        "choice 1** more strongly suggests" in string_response_lower or \
        "choice 1" in string_response_lower and "choice 2" not in string_response_lower: 
        result = 1
        # print(f"Parsed response with label 1: {string_response}")
    elif "Choice 2 more strongly suggests" in string_response or \
        "choice 2** more strongly suggests" in string_response_lower or \
        "choice 2" in string_response_lower and "choice 1" not in string_response_lower: 
        result = 0
        # print(f"Parsed response with label 0: {string_response}")
    elif prompt_method == "zero_shot" or (client is None): # will not re-query for zero-shot responses. 
        result = 0.5
        unparsed = 1
        # print(f"Did not parse: {string_response}")
    else:
        # print(f"Did not parse: {string_response}, re-querying...")
        # re-query the same LLM
        re_queried = True
        if prompt_method == "positive":
            re_query_prompt = prompts.re_query_prompt_positive
        else:
            re_query_prompt = prompts.re_query_prompt_negative

        re_query_prompt = re_query_prompt.format(prev_response = string_response, X = target_item)
        re_query_completion = make_chat_call(client, model, re_query_prompt, 10)

        if "gpt" in model:
            re_query_response = re_query_completion.choices[0].message.content.strip()
        
        if "1" in re_query_response and "2" not in re_query_response:
            result = 1
            # print(f"Re-queried response with label 1: {re_query_response}")
        elif "2" in re_query_response and "1" not in re_query_response:
            result = 0
            # print(f"Re-queried response with label 0: {re_query_response}")
        elif "tie" in re_query_response.lower():
            result = 0.5
        else: 
            print(f"Did not parse: {string_response}")
            result = None
            unparsed = 1
        
    # get the cost
    if re_queried:
        prompt_cost = 0.0025
        completion_cost = 0.0075
        total_prompt_tokens = re_query_completion.usage.prompt_tokens
        total_completion_tokens = re_query_completion.usage.completion_tokens
    
        cost = (prompt_cost * total_prompt_tokens + completion_cost * total_completion_tokens) / 1000
    else:
        cost = 0
    
    return result, cost, unparsed


def batch_results(args, batch_path, result_string, save_substring, client):

    # create structs to support finding what was run

    # list of dims total_completions, num_choices, num_choices
    results = [[[None for i in range(args.num_choices)] for j in range(args.num_choices)] for k in range(args.total_completions)]
    found_results = np.zeros((args.total_completions, args.num_choices, args.num_choices))

    # load the output jsonl
    with open(batch_path + result_string, "r") as f:
        lines = f.readlines()
    
    prompt_tokens, completion_tokens = 0, 0
    
    # parse the custom_id field
    for line in lines:
        result = json.loads(line)
        indices = result["custom_id"].split("_")
        c_index = int(indices[0])
        i = int(indices[1])
        j = int(indices[2])
        if i > j: raise ValueError("i > j")

        found_results[c_index, i, j] = 1
        results[c_index][i][j] = result["response"]["body"]["choices"][0]["message"]["content"]
        prompt_tokens += result["response"]["body"]["usage"]["prompt_tokens"]
        completion_tokens += result["response"]["body"]["usage"]["completion_tokens"]
    
    # gpt-4o
    prompt_cost = 0.0025
    completion_cost = 0.0075
    first_pass_cost = (prompt_cost * prompt_tokens + completion_cost * completion_tokens) / 1000
    print(f"Prompt tokens used: {prompt_tokens}, Completion tokens used: {completion_tokens}, total cost for first pass: {first_pass_cost}")

    # put the results in the save_dict
    unparsed = 0
    requery_cost = 0
    requery_counts = 0
    for c_index in range(args.total_completions):

        # load the save_dict to get the prompt
        save_path = batch_path + str(c_index) + "/"
        save_file = None
        for file in os.listdir(save_path):
            if type(save_substring) == list:
                for sub in save_substring:
                    if sub in file:
                        if save_file is not None: raise ValueError("Multiple files in save path, please specify. ")
                        save_file = file
            elif type(save_substring) == str:
                if save_substring in file:
                    if save_file is not None: raise ValueError("Multiple files in save path, please specify. ")
                    save_file = file
        
        if save_file is None: raise ValueError("No save file found")
        with open(save_path + save_file, "r") as f:
            save_dict = json.load(f)
        
        # identify the items that need to be updated
        for i in range(args.num_choices):
            for j in range(i+1, args.num_choices):
                if found_results[c_index, i, j] == 1:

                    # get the item from the save_dict
                    item = save_dict[i][j]
                    if (item["indices"][0] != i) or (item["indices"][1] != j): raise ValueError(f"Indices do not match, {item['indices']} != {(i, j)}")
                    if item["response"] is not None: raise ValueError("Response already exists")

                    item["response"] = results[c_index][i][j]
                    parsed = parse_batch_response(results[c_index][i][j], args.model, args.prompt_method, item["target_item"], client)
                    item["parsed"] = parsed[0]
                    requery_cost += parsed[1]
                    unparsed += parsed[2]
                    if parsed[1] > 0: requery_counts += 1
                    save_dict[i][j] = item
            
        # save the updated save_dict
        now = datetime.now()
        date_string = now.strftime("%m-%d-%H-%M-%S_")
        if client is None:
            output_file = save_path + date_string + "save_queried.json"
        else:
            output_file = save_path + date_string + "save_requeried.json"
        with open(output_file, "w") as f:
            json.dump(save_dict, f)
        print(f"save dict saved to {output_file}")

        print(f"Re-query cost: {requery_cost}, number of re-queries: {requery_counts}")
        print(f"Unparsed responses: {unparsed}/{len(lines)}")


def send_batch(client, batch_path):
    batch = client.files.create(
        file=open(batch_path, 'rb'),
        purpose="batch"
    )
    batch_id = batch.id
    print(f"Batch file created with id {batch_id}")

    batch_object = client.batches.create(
        input_file_id=batch_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "inverse model"}
    )
    print(f"Batch object created:\n{batch_object}")


def check_batch(client, batch_id):
    print(f"Checking batch with id {batch_id}")
    batch_status = client.batches.retrieve(batch_id)
    print(f"Batch status:\n{batch_status}")