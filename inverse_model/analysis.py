import os, sys, json, argparse, numpy as np, matplotlib.pyplot as plt
from scipy.stats import spearmanr

def get_numbers_list(s):
    numbers_string = s.split("_")[0]
    numbers_list_of_string = numbers_string.split("-")
    numbers_list = [int(x) for x in numbers_list_of_string]
    return numbers_list

def get_numbers_lists(path):
    numbers_lists = []
    paths = []
    for file in os.listdir(path):
        if "save" in file:
            numbers_list = get_numbers_list(file)
            numbers_lists.append(numbers_list)
            paths.append(path + file)
    return numbers_lists, paths

def get_largest_numbers_path(path):
    numbers_lists, paths = get_numbers_lists(path)
    candidates = [1] * len(numbers_lists)
    winner_index = get_largest_numbers_list(numbers_lists, candidates, 0)
    return paths[winner_index]

def get_largest_numbers_list(numbers_lists, candidates, index):
    max_index_num = -1

    # print(numbers_lists)
    # print(candidates)

    for i in range(len(numbers_lists)):
        numbers_list = numbers_lists[i]
        if candidates[i] == 0:
            continue
        if numbers_list[index] > max_index_num:
            max_index_num = numbers_list[index]
    
    # print(max_index_num)
    
    for i in range(len(numbers_lists)):
        if candidates[i] == 0:
            continue
        numbers_list = numbers_lists[i]
        if numbers_list[index] != max_index_num:
            candidates[i] = 0
    
    if sum(candidates) == 1:
        winner_index = candidates.index(1)
        return winner_index
    elif index == 4:
        raise ValueError("Error: multiple candidates for largest numbers list")

    return get_largest_numbers_list(numbers_lists, candidates, index + 1)


def construct_matrix_from_save_dict(args):
    ranking_matrix = np.zeros((args.total_completions, args.num_choices, args.num_choices))

    for c_index in range(args.total_completions):

        # get the correct save dict
        path = args.path + args.file_prefix + str(c_index) + "/"
        largest_numbers_path = get_largest_numbers_path(path)
        with open(largest_numbers_path) as f:
            saved_data = json.load(f)

        for i in range(len(saved_data)):
            for j in range(i+1, len(saved_data)):
                value = saved_data[i][j]["parsed"]
                flip = saved_data[i][j]["shuffled_ordering"]
                
                if flip == [1, 0]:
                    ranking_matrix[c_index][j][i] = value
                    ranking_matrix[c_index][i][j] = 1 - value
                else:
                    ranking_matrix[c_index][i][j] = value
                    ranking_matrix[c_index][j][i] = 1 - value
        
        # save the ranking matrix
        np.save(path + "ranking_matrix", ranking_matrix)
        print(f"Created ranking matrix for completion {c_index}")


def load_data(args):
    num_completions = args.total_completions
    print(f"using {num_completions} completions.")
    if args.debug and len(os.listdir(args.path)) < num_completions:
        raise ValueError(f"Number of completions available {len(os.listdir(args.path))} is less than specified")

    total_ranking_matrix = np.zeros((num_completions, args.num_choices, args.num_choices))
    for i in range(num_completions): # number of iterations
        # if args.debug:
        #     print("Loading completion", i)
        path = args.path + args.file_prefix + str(i) + "/"
        for file in os.listdir(path):
            if "ranking_matrix" in file:
                ranking_matrix = np.load(path + file)
                # print(ranking_matrix[i])
                # ranking_matrix dims (num_completions, num_choices, num_choices)

                if args.debug:
                    # checking shape match for direct addition
                    if ranking_matrix.shape[1] != total_ranking_matrix.shape[1] or \
                        ranking_matrix.shape[2] != total_ranking_matrix.shape[2]:
                        print(f'ranking_matrix dims 1 and 2 {ranking_matrix.shape[1:]} unequal to total_ranking_matrix dims 1 and 2 {total_ranking_matrix.shape[1:]}')
                        raise ValueError(f"Ranking matrix shape {ranking_matrix.shape} does not match total shape {total_ranking_matrix.shape}")

                    if ranking_matrix.shape[0] > num_completions:
                        print(f'detected only {num_completions} completions, but ranking_matrix intended for {ranking_matrix.shape[0]} completions, running partial completion analysis. ')

                    # if any entries in array total_ranking_matrix[i] are nonzero when they should be
                    for j in range(i, num_completions):
                        if np.sum(total_ranking_matrix[j]) > 0:
                            print(total_ranking_matrix[0, 0], end="\n\n")
                            raise ValueError(f"Ranking matrix already contains data for completion {j} when loading data from completion {i}")
                    
                    # if any entries outside ranking_matrix[i] are nonzero
                    for j in range(i+1, ranking_matrix.shape[0]):
                        if np.sum(ranking_matrix[j]) > 0:
                            print(f'path: {path}, completion {i}, checking index {j}, matrix: {ranking_matrix[j]}')
                            raise ValueError(f"individual ranking matrix should only contain data for completion up to {i}, but instead also contains data for completion {j}")

                    # check that previous entries match the ones existing
                    # for j in range(i):
                    #     if (not np.array_equal(ranking_matrix[j], total_ranking_matrix[j])) and \
                    #        (not np.array_equal(ranking_matrix[j], np.zeros((args.num_choices, args.num_choices)))):
                    #         print(f'path: {path}, completion {i}, checking index {j}')
                    #         print(f'total {(total_ranking_matrix[j] - ranking_matrix[j])}')
                            # debugging: so it is indeed that one is requeried saved and one is not. 
                            # it looks like the errors are in twos or threes - just like the groups that I originally queried in
                            # but those ran through the analysis fine by themselves; thus this requerying process 
                            # is doing something during the modification of the old matrix to the new matrix, 
                            # that somehow isn't properly zeroing them out. 

                            # raise ValueError(f"individual ranking matrix does not match total ranking matrix for completion {j}")

                total_ranking_matrix[i] = ranking_matrix[i]

            # elif "save.json" in file:
            #     saved_data = json.load(open(path + file))
                # saved_data 2d-list, dims (num_choices, num_choices), each element is a dict
                # keys: "indices", "completion_index", "prompt", "response", "parsed", "shuffled_ordering", 
                #       "target_item", "choices", "choices_context", "shuffled_choices_context_1", 
                #       "shuffled_choices_context_2", "prompt_type", "prompt_method"
    
    return total_ranking_matrix, num_completions #, saved_data


def transform_pairwise_rankings_into_average_rankings(args, total_ranking_matrix, num_completions):
    # input: total_ranking_matrix (num_completions, num_choices, num_choices)
    # output: average_rankings (num_completions, num_choices)

    summed_pairwise_rankings = np.sum(total_ranking_matrix, axis=2) # sum over comparisons
    # print(summed_pairwise_rankings)
    average_rankings = np.zeros((num_completions, args.num_choices))

    # turn something like [1, 4, 5, 5] (pairwise comparison sum) into [1, 2, 3.5, 3.5] (ranking)
    # not necessarily ordered; could be [4, 5, 1, 5] into [2, 3.5, 1, 3.5]

    for i in range(num_completions):
        # print(summed_pairwise_rankings[i])
        completion_rankings = summed_pairwise_rankings[i]
        current_rank = 1
        while current_rank <= args.num_choices:
            min_value = np.min(completion_rankings)
            # print(min_value)
            count_min_value = np.sum(completion_rankings == min_value)
            value_to_assign = current_rank + (count_min_value - 1) / 2
            # assigns 1 -> 1, (1,2) -> 1.5

            for j in range(args.num_choices):
                if completion_rankings[j] == min_value:
                    average_rankings[i][j] = value_to_assign
                    completion_rankings[j] = 10000

            current_rank += count_min_value
        # print(average_rankings[i])
        
    return average_rankings


def analysis(args):
    overall_ranking_matrix = np.zeros((args.num_choices, args.num_choices))
    for i in range(len(os.listdir(args.load_previous)) - 1): # number of iterations, DS_Store
        path = args.load_previous + str(i) + "/"
        for file in os.listdir(path):
            if "ranking_matrix" in file:
                ranking_matrix = np.load(path + file)
            elif "save.json" in file:
                saved_data = json.load(open(path + file))
    
        overall_ranking_matrix += ranking_matrix

    # analysis option 1: get the ranking of the choices
    sums = np.sum(overall_ranking_matrix, axis=1) # stronger evidence has a higher sum
    
    # x values are the average rankings made by the LLM, I have these in sums but need to get the averaging right. 
    # y values are the average rankings made by humans, I don't have these. 
    # then I plot x with y and see the spearman correlation. This gives me figure 3 of the Jern paper.

    ranking = np.argsort(sums) # weaker evidence should be first, thus perfect match is 0 1 2... for unshuffled choices
    print(f"Ranking of choices: {ranking}")

    # analysis option 2: pairwise accuracy
    correct_density = 0
    densities = []
    for i in range(args.num_choices):
        for j in range(i+1, args.num_choices):
            for c in range(ranking_matrix.shape[0]):
                pair_correct_density += 1 - saved_data[c][i][j]["parsed"]
            
            pair_correct_density /= ranking_matrix.shape[0] # divide by number of completions
            correct_density += pair_correct_density
            densities.append(pair_correct_density)
    
    print(f"Pairwise accuracy: {correct_density / (args.num_choices * (args.num_choices - 1) / 2)}")

    # plot the sure-ness of models (0 and 1 are sure, between is less sure)
    count = 0
    for density in densities:
        if density != 0 and density != 1:
            count += 1 
    print(count / len(densities))
    # plt.hist(densities, bins=11)
    # plt.show()

    # print the amount of comparisons won by each choice
    # for choice_indices, score in zip(range(args.num_choices), sums):
    #     score = round(score, 1)
    #     print(f"Choice {choice_indices} won {score} comparisons")


    # sample some of the ones where the LLM answers incorrectly
    # sample_amount = 10
    # counter = 0
    # for i in range(args.num_choices):
    #     for j in range(i+2, args.num_choices):
    #         if saved_data[str((i,j))]["parsed"] > 0.5:
    #             print(f"Choice {i} vs {j}: {saved_data[str((i,j))]}")
                
    #             counter += 1
    #             if counter >= sample_amount:
    #                 return


def flip_decision_string(s):
    if s == "dcax/bcax": return "cbax/dbax" # inconsistencies
    if s == "bad/bax": return "bax/bac"

    items = s.split("/")
    reversed_items = items[::-1]
    return ("/").join(reversed_items)


def preprocess_model_rankings(args, rankings_raw):
# in the raw file, the format is:
# {decision choice, with the selected choice last} {float score}

    choices = {} # a mapping from representing string to index. dcbax -> 0, cbax -> 1, etc. 
    path = f"src/inverse_model/{args.prompt_type}_weak_to_strong.txt"
    if args.prompt_type == 'negative':
        path = f"src/inverse_model/{args.prompt_type}_weak_to_strong_fixed.txt"
    with open(path) as f:
        lines = f.readlines()
        for i in range(len(lines)): 
            line = lines[i]
            choices[line.strip()] = i

    rankings = [0] * 47 
    for line in rankings_raw:
        separated = line.split(" ")
        decision_string = separated[0]
        flipped_decision_string = flip_decision_string(decision_string) # representing strings
        correct_index = choices[flipped_decision_string]

        float_string = separated[1]
        float_value = float(float_string)

        if args.debug:
            if rankings[correct_index] != 0:
                raise ValueError("previous value nonzero")
        
        rankings[correct_index] = float_value
    return rankings


def pair_corr(ranking1, ranking2, name1, name2):

    # spearman correlation
    print(f'spearman correlation between {name1} and {name2}: {spearmanr(ranking1, ranking2)}')
    
    # plot the data
    plt.scatter(ranking1, ranking2)
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.title(f"{name1} vs. {name2} for {args.prompt_type} stimuli")
    # plot a line on the diagonal
    plt.plot([0, 47], [0, 47], color="gray")
    path = f"src/inverse_model/plots/{args.model}/{args.prompt_type}/{args.prompt_method}/"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f"{path}{name1}_{name2}.png")
    plt.close()

    # print('plot saved.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_completions", type=int)
    parser.add_argument("--prompt_method", type=str, default="zero_shot")
    parser.add_argument("--debug", type=str, choices=["True", "False"])
    parser.add_argument("--num_choices", type=int, default=47)
    parser.add_argument("--prompt_type", type=str, default="positive")
    parser.add_argument("--model", type=str)
    parser.add_argument("--construct_matrix_from_save_dict", type=str, default="False")
    parser.add_argument("--path", type=str)
    parser.add_argument("--file_prefix", type=str, default="")
    args = parser.parse_args()
    
    args.debug = True if args.debug == "True" else False
    args.construct_matrix_from_save_dict = True if args.construct_matrix_from_save_dict == "True" else False
    return args


if __name__ == "__main__":
    args = get_args()
    np.set_printoptions(threshold=sys.maxsize)

    # if there is no ranking matrix, construct it from the save dict
    if args.construct_matrix_from_save_dict:
        construct_matrix_from_save_dict(args)

    # get the LLM data from the ranking matrix
    total_ranking_matrix, num_completions = load_data(args)
    individual_rankings = transform_pairwise_rankings_into_average_rankings(args, total_ranking_matrix, num_completions)
    # individual_rankings is like #participants x #choices
    average_rankings = np.mean(individual_rankings, axis=0) # average positional rankings
    # print(average_rankings)

    # run variance analysis on individual rankings
    std_devs = []
    for i in range(individual_rankings.shape[1]):
        # print(f"Choice {i}: {individual_rankings[:, i]}")
        # print(f"Choice {i} std dev: {np.std(individual_rankings[:, i])}")
        std_devs.append(np.std(individual_rankings[:, i]))
    print(f"Average std dev: {np.mean(std_devs)}")
    
    # raise ValueError("std analysis complete.")

    # get the human data
    with open(f"src/inverse_model/human_results_{args.prompt_type}.txt") as f:
        human_rankings = f.readlines()
    human_rankings = [float(x.strip()) for x in human_rankings]
    # print(human_rankings)

    # get the model data
    models = ["absolute_utility", "relative_utility", "likelihood", "marginal_likelihood"]
    model_names = ["Absolute Utility", "Relative Utility", "Likelihood", "Marginal Likelihood"]
    model_rankings = []
    for model in models:
        with open(f"src/inverse_model/model_results/{model}_{args.prompt_type[:3]}.txt") as f:
            model_rankings_raw = f.readlines()
        model_ranking = preprocess_model_rankings(args, model_rankings_raw)
        model_rankings.append(model_ranking)

        # print(model, model_ranking)
    
    # human vs. LLM
    pair_corr(human_rankings, average_rankings, "human", "LLM")

    # absolute utility vs. LLM
    pair_corr(model_rankings[0], average_rankings, model_names[0], "LLM")
    # relative utility vs. LLM
    pair_corr(model_rankings[1], average_rankings, model_names[1], "LLM")
    # likelihood vs. LLM
    pair_corr(model_rankings[2], average_rankings, model_names[2], "LLM")
    # marginal likelihood vs. LLM
    pair_corr(model_rankings[3], average_rankings, model_names[3], "LLM")

    if args.debug:
        print("\n")
        pair_corr(human_rankings, model_rankings[0], "human", models[0])
        pair_corr(human_rankings, model_rankings[1], "human", models[1])
        pair_corr(human_rankings, model_rankings[2], "human", models[2])
        pair_corr(human_rankings, model_rankings[3], "human", models[3])



def test_transform():
    input = [[1], [1], [1]]
    args = argparse.Namespace(num_choices=3)
    formatted_input = np.array([input])
    expected_output = np.array([[2, 2, 2]])

    output = transform_pairwise_rankings_into_average_rankings(args, formatted_input, 1)
    print(output)
    assert(np.array_equal(output, expected_output))

    input = [[1], [4], [5], [5]]
    args = argparse.Namespace(num_choices=4)
    formatted_input = np.array([input])
    expected_output = np.array([[1, 2, 3.5, 3.5]])
    
    output = transform_pairwise_rankings_into_average_rankings(args, formatted_input, 1)
    print(output)
    assert(np.array_equal(output, expected_output))

    input = [[4], [5], [1], [5]]
    args = argparse.Namespace(num_choices=4)
    formatted_input = np.array([input])
    expected_output = np.array([[2, 3.5, 1, 3.5]])

    output = transform_pairwise_rankings_into_average_rankings(args, formatted_input, 1)
    print(output)
    assert(np.array_equal(output, expected_output))

    print("All tests passed.")

# test_transform()