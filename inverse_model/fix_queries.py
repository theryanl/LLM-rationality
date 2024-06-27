import os, json, numpy as np, time
from datetime import datetime
from src.inverse_model.utils import *

def replace_results(args, fix):
    # partially fixed versions (replaced) will be stored in new_dir_path
    new_dir_path = "src/inverse_model/results/pairwise/negative/cot_replaced/"
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    dir_path = "src/inverse_model/results/pairwise/negative/cot/"
    # where all the results we need to fix are

    for i in range(42):
        path = f'{dir_path}{i}' # 41 completions
        
        for file in os.listdir(path):
            if 'ranking_matrix' in file:
                # it is the ranking matrix, stored in npy format
                ranking_matrix = np.load(f'{path}/{file}') 
                # shape is num_completions, num_choices, num_choices
                # for the change, I should change the dims for all completions. 
                # I should fill with np.nan so that it is easily noticeable
                # for each completion (shape num_choices x num_choices), 
                # each pair is originally queried once and mirrored with the opposite value
                # I need to replace both the row and the column. 
                # - X A X B X
                # X - A X B X
                # A A - A ? A
                # x X A - B X
                # B B ? B - B
                # X X A X B X
                # Ok, this is more work than it is worth. let's just not replace. 



            elif 'save' in file:
                # it is the save file, stored in json format
                with open(f'{path}/{file}', 'r') as f:
                    save = json.load(f)


def requery_results(args, client, fixes):
    # Do a similar thing to pairwise_experiment() in inverse_model.py, 
    #   but only requery for one index (fix), where we were intially wrong. 
    # fixes: list of dict with keys:
    #       index: index in the 47 choices that was problematic and needs requerying
    #       incorrect: string representing what was previously incorrectly queried. 
    #                  format is "x/ba/dc".
    #       correct: string representing what should actually be queried. same format. 

    start_time = time.time()
    fixed_options_path = "src/inverse_model/negative_weak_to_strong_fixed.txt"
    choices = get_choices(args.num_choices, fixed_options_path)

    # options_path = "src/inverse_model/negative_weak_to_strong.txt"
    # choices_old = get_choices(args.num_choices, options_path)
    # confirmed to be correct

    # create pairs that need to be replaced
    pairs = []
    fix_indices = [fix['index'] for fix in fixes]
    for fix_index in fix_indices: # loop over all fix indices
        for j in range(args.num_choices): # loop over all normal choices for comparison

            if j in fix_indices and fix_index >= j:
                continue
                # we don't want to query (28, 28), or both (28, 29) and (29, 28)

            pair = dict()
            pair["choices"] = [choices[fix_index], choices[j]]
            pair["indices"] = (fix_index, j)
            pairs.append(pair)

    total_cost = 0
    max_tokens = 500 if args.prompt_method == "cot" else 50

    # create a location to save fixed results
    new_dir_path = "src/inverse_model/results/pairwise/negative/cot_fixed/"
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
    
    dir_path = "src/inverse_model/results/pairwise/negative/cot/"
    # where all the results we need to fix are

    for c_index in range(args.c_index_start, args.c_index_start + args.num_completions):
        print(f"Completion {c_index}")
        
        path = f'{dir_path}{c_index}' # 41 completions

        # load the previous ranking matrices and save dicts inside the completions loop
        for file in os.listdir(path):
            if 'ranking_matrix' in file:
                ranking_matrix = np.load(f'{path}/{file}') 
                # shape is num_completions, num_choices, num_choices

            elif 'save' in file:
                with open(f'{path}/{file}', 'r') as f:
                    save = json.load(f)
        
        # next: move to assign_context for each pair
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
        # prompts look good, 220 of them total: 5 * 47 - 5choose4 - 5
        # print(len(prompts))
        # print(prompts[3])

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
            
            ranking_matrix[c_index, indices[0], indices[1]] = result
            ranking_matrix[c_index, indices[1], indices[0]] = 1 - result
        
            save[indices[0]][indices[1]] = {"indices": indices, 
                                            "completion_index": c_index, 
                                            "prompt": prompt_text, 
                                            "response": response_text, 
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
        
        # zero out dims other than the index
        for i in range(args.total_completions): # ah, here is the bug. I had num_completions here, not 
                                              # total_completions. This is why it didn't cause an issue 
                                              # for 17 and under. because it only removed to zero up till 17. 
                                              # so it's just the old ones previously, that were still saved together
                                              # since they were queried together and the matrix was shared. no biggie 
                                              # as long as we construct total_ranking_matrix correctly in the analysis step. 
            if i != c_index:
                # set all the values to 0
                ranking_matrix[i] = 0
            
        # save the results
        # get the time for the filename
        now = datetime.now()
        date_string = now.strftime("%m-%d-%H-%M-%S_")
        if not os.path.exists(new_dir_path + str(c_index)+ "/"):
            os.makedirs(new_dir_path + str(c_index) + "/")

        output_path = new_dir_path + str(c_index) + "/" + date_string
        output_file = output_path + "ranking_matrix.npy"
        np.save(output_file, ranking_matrix)
        print(f"Ranking matrix saved to {output_file}")

        output_file = output_path + "save.json"
        with open(output_file, 'w') as f:
            json.dump(save, f)
        print(f"Results saved to {output_file}")
        print(f"Total cost: {total_cost}")
        print(f"Completed completion {c_index}, {time.time() - start_time} seconds")

