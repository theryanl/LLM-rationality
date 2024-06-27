import src.inverse_model.prompts as prompts
import random, copy, os, time
from openai import AzureOpenAI
import anthropic

def get_choices(num_choices, path):
# Input: 
#   num_choices: int, the number of choices to get
#
# Returns: 
#   a list of choices with length num_choices
#   each choice is a list of options
#   each option is a list of items
    
    with open(path, "r") as f:
        lines = f.read().split("\n")
    
    choices = []
    for i in range(len(lines)):
        line = lines[i]
        choice_options = line.split("/") # ["ax", "b", "c"]
        for j in range(len(choice_options)):
            choice_options[j] = [*choice_options[j]] # [["a", "x"], ["b"], ["c"]]
        choices.append(choice_options)

    # instead of randomly shuffling, want to randomly select while preserving the order
    sampled_indices = random.sample(range(len(choices)), num_choices)
    sampled_indices.sort()
    choices = [choices[i] for i in sampled_indices]
    return choices


def assign_context(pair, prompt_type):
# Input: 
#   pair: dict, key "choices" is a list of two choices (see choice format in get_choices)
#                                          or more choices when doing the full ranking at once
#   prompt_type: string, "positive" or "negative"
#
# Returns:
#   pair: dict, keys "choices", "choices_context", "mapping", "target_item"
#   "choices": two (or more) choices associated with the pair, where each choice is of the form: [[a, b], [c], [x]]
#   "choices_context": same as "choices", but with the context assigned to the items, e.g., [["brown", "yellow"], ["red"], ["black"]]
#   "mapping": dict, mapping of symbols to context, e.g., {"a": "brown", "b": "yellow", "c": "red", "d": "blue", "x": "black"}
#   "target_item": string, the context assigned to the target item, e.g., "black"
# 
# Notes: 
#   positive context maps to candy colors, negative context maps to shock locations
#   mapping of a/b/c/d/x to context (e.g. colors) is assigned randomly

    # get the context, i.e., colors/shock locations
    if prompt_type == "positive":
        context = prompts.positive_context
    else:
        context = prompts.negative_context
    random.shuffle(context)

    # assign context to symbols, i.e., "a" = "brown"
    symbols = ["a", "b", "c", "d", "x"]
    mapping = dict()
    for i in range(5):
        mapping[symbols[i]] = context[i]
    
    # across all choices in the comparison "pair", the mapping is the same.
    # pair can have more than two choices when doing ranking_experiment. 
    # store these associated with the "pair". 
    pair["target_item"] = mapping["x"]
    pair["mapping"] = mapping
    
    # create a new "choices_context", with the context assigned to the choices
    pair["choices_context"] = copy.deepcopy(pair["choices"])

    # for each choice, and each option (group of items), and each item, assign the context to the items. 
    for choice in pair["choices_context"]:
        for option in choice:
            for i in range(len(option)):
                option[i] = mapping[option[i]]
    
    return pair


class ShuffledChoice:
    def __init__(self, choice):
        self.choice = choice
        self.shuffled_choice = None
        self.shuffled_ordering = None
        self.chosen_option_index = 0

    def shuffle(self):
        if self.shuffled_choice is not None:
            raise ValueError("Choice already shuffled")
        
        new_ordering = list(range(len(self.choice)))
        random.shuffle(new_ordering)
        self.shuffled_ordering = new_ordering
        self.shuffled_choice = [self.choice[i] for i in new_ordering]
        self.chosen_option_index = self.shuffled_ordering.index(self.chosen_option_index)

    def shuffle_items(self):
        if self.shuffled_choice is None:
            raise ValueError("Shuffle the choice first")

        for choice in self.shuffled_choice:
            random.shuffle(choice)

    def get_chosen_option_index(self):
        return self.chosen_option_index


def shuffle_context(contextualized_choices):
# Input:
#   contextualized_choices: 
#   dict with keys "choices" : list of original choices, where each choice is of the form: [[a, b], [c], [x]]
#                  "choices_context": list of choices with context assigned to items, e.g., [["brown", "yellow"], ["red"], ["black"]]
#                  "mapping": dict, mapping of symbols to context, e.g., {"a": "brown", "b": "yellow", "c": "red", "d": "blue", "x": "black"}
#                  "target_item": string, the context assigned to the target item, e.g., "black"
# 
# Returns:
#   contextualized_choices: new keys "shuffled_choices_context", "shuffled_ordering
#       "shuffled_choices_context": list of ShuffledChoice objects
#       "shuffled_ordering": list of indices, mapping from the original order to the new order of all the choices in the comparison. 
#   the order of the choices is shuffled
#   the order of the options in each choice is shuffled
#   the order of the items in each option is shuffled

    # 1. shuffling within the choices
    shuffled_choices = []
    for choice in contextualized_choices["choices_context"]:
        shuffled_choice = ShuffledChoice(choice)
        shuffled_choice.shuffle()
        shuffled_choice.shuffle_items()
        shuffled_choices.append(shuffled_choice)
    
    # 2. shuffling the order of the choices
    # first, get the mapping from the original order to the new order
    new_ordering = list(range(len(shuffled_choices))) # this is just [0, 1]
    random.shuffle(new_ordering)
    
    # then, shuffle the choices
    shuffled_choices_context = [shuffled_choices[i] for i in new_ordering]
    contextualized_choices['shuffled_choices_context'] = shuffled_choices_context
    contextualized_choices['shuffled_ordering'] = new_ordering
    # note this is from old to new, indices of the list are the old indices. 
    # For instance, if new_ordering is [5, ...], then the 1st choice in the new list is the 6th choice in the old list.
        
    return contextualized_choices


def construct_choice_description(choice, prompt_type):
# Input:
#   choice: list of options, e.g. [["black", "yellow"], ["brown", "blue"], ["red"]]
#
# Returns:
#  a string describing the option
    
    if prompt_type == "positive":
        option_title = "Bag"
    else:
        option_title = "Set"

    choice_description = ""
    for i in range(len(choice)):
        option = choice[i]
        option_description = f"{option_title} {i+1}: "
        for j in range(len(option)):
            item = option[j]
            if j == 0:
                option_description += f"{item}"
            else:
                option_description += f", {item}"
        choice_description += option_description + ". \n"
    return choice_description


def construct_prompt(pair, prompt_type, prompt_method):
# Input:
#   pair: see return of shuffle_context
#   prompt_type: string, "positive" or "negative"
#
# Returns:
#   a string prompt to query the LLM
    
    if prompt_method == "cot":
        prompt_method_text = prompts.cot_prompt
    else:
        prompt_method_text = prompts.zero_shot_prompt

    choice_1 = pair["shuffled_choices_context"][0] # ShuffledChoice object
    choice_2 = pair["shuffled_choices_context"][1]

    if prompt_type == "positive":
        prompt = prompts.positive_instructions
        if len(choice_1.shuffled_choice) == 1: bag_choice_1 = prompts.bag_choice_single
        else: bag_choice_1 = prompts.bag_choice_multiple

        if len(choice_2.shuffled_choice) == 1: bag_choice_2 = prompts.bag_choice_single
        else: bag_choice_2 = prompts.bag_choice_multiple
        
        prompt = prompt.format(
            bag_choice_1 = bag_choice_1,
            choice_1_bags = construct_choice_description(choice_1.shuffled_choice, prompt_type),
            choice_1_chosen_bag = choice_1.get_chosen_option_index() + 1,
            bag_choice_2 = bag_choice_2,
            choice_2_bags = construct_choice_description(choice_2.shuffled_choice, prompt_type),
            choice_2_chosen_bag = choice_2.get_chosen_option_index() + 1,
            X = pair["target_item"],
            prompt_method = prompt_method_text
        )
    else:
        prompt = prompts.negative_instructions
        if len(choice_1.shuffled_choice) == 1: shock_choice_1 = prompts.shock_choice_single
        else: shock_choice_1 = prompts.shock_choice_multiple

        if len(choice_2.shuffled_choice) == 1: shock_choice_2 = prompts.shock_choice_single
        else: shock_choice_2 = prompts.shock_choice_multiple

        prompt = prompt.format(
            shock_choice_1 = shock_choice_1,
            choice_1_sets = construct_choice_description(choice_1.shuffled_choice, prompt_type),
            choice_1_chosen_set = choice_1.get_chosen_option_index() + 1,
            shock_choice_2 = shock_choice_2,
            choice_2_sets = construct_choice_description(choice_2.shuffled_choice, prompt_type),
            choice_2_chosen_set = choice_2.get_chosen_option_index() + 1,
            X = pair["target_item"],
            prompt_method = prompt_method_text
        )

    return pair["indices"], prompt


def construct_ranking_prompt(contextualized_choices, prompt_type, prompt_method):

    if prompt_method == "cot":
        prompt_method_text = prompts.cot_prompt
    else:
        prompt_method_text = ""
    
    if prompt_type == "positive":
        prompt = prompts.positive_ranking_instructions
        choices_list = ""
        for i in range(len(contextualized_choices["shuffled_choices_context"])): # should be 47
            choice = contextualized_choices["shuffled_choices_context"][i] # this is a ShuffledChoice object
            choices_list += prompts.choices_list.format(
                index = i+1, 
                choice_bags = construct_choice_description(choice.shuffled_choice, prompt_type),
                chosen_bag = choice.get_chosen_option_index() + 1
            )

        prompt = prompt.format(
            choices_list = choices_list,
            X = contextualized_choices["target_item"],
            prompt_method = prompt_method_text
        )
    else:
        raise ValueError("Negative ranking not implemented yet")
    
    return prompt

def make_chat_call(client, model, message, max_tokens):

    if "gpt" in model:
        full_prompt = [{'role': 'user', 'content': message}]
    elif "claude" in model:
        full_prompt = [{'role': 'user', 'content': [{"type": "text", "text": message}]}]

    response = None
    wait_time = 5
    while response is None:
        try:
            if "gpt" in model:
                response = client.chat.completions.create(
                    model=model,
                    messages=full_prompt,
                    max_tokens=max_tokens,
                    n=1
                )
                for r in response.choices:
                    if r.finish_reason == "content_filter":
                        print("Content filter triggered, trying again")
                        response = None
            
            elif "claude" in model:
                response = client.messages.create(
                    model=model,
                    messages=full_prompt,
                    max_tokens=max_tokens,
                    temperature=1
                )
                if response.stop_reason != "end_turn":
                    print("End turn not triggered, trying again")
                    response = None
                
                # print(response.content)
            
        except Exception as e:
            print(f'Caught exception {e}.')
            # print(f'Waiting {wait_time} seconds.')
            time.sleep(wait_time)

    return response


def parse_response(response, model, prompt_method, target_item, client):
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

    if "gpt" in model: 
        string_response = response.choices[0].message.content.strip()
    elif "claude" in model:
        string_response = response.content[0].text.strip()

    string_response_lower = string_response.lower()

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
    elif prompt_method == "zero_shot": # will not re-query for zero-shot responses. 
        result = 0.5
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
        elif "claude" in model:
            re_query_response = re_query_completion.content[0].text.strip()
        
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
        
    # get the cost
    if model == "gpt-4-1106-preview" or model == "gpt-4-0125-preview":
        prompt_cost = 0.01
        completion_cost = 0.03

        if re_queried:
            total_prompt_tokens = response.usage.prompt_tokens + re_query_completion.usage.prompt_tokens
            total_completion_tokens = response.usage.completion_tokens + re_query_completion.usage.completion_tokens
        else: 
            total_prompt_tokens = response.usage.prompt_tokens
            total_completion_tokens = response.usage.completion_tokens
        
    elif "claude-3-opus" in model:
        prompt_cost = 0.015
        completion_cost = 0.075

        if re_queried:
            total_prompt_tokens = response.usage.input_tokens + re_query_completion.usage.input_tokens
            total_completion_tokens = response.usage.output_tokens + re_query_completion.usage.output_tokens
        else: 
            total_prompt_tokens = response.usage.input_tokens
            total_completion_tokens = response.usage.output_tokens
    
    elif "gpt-4o" in model:
        prompt_cost = 0.0025
        completion_cost = 0.0075

        if re_queried:
            total_prompt_tokens = response.usage.prompt_tokens + re_query_completion.usage.prompt_tokens
            total_completion_tokens = response.usage.completion_tokens + re_query_completion.usage.completion_tokens
        else:
            total_prompt_tokens = response.usage.prompt_tokens
            total_completion_tokens = response.usage.completion_tokens

    cost = (prompt_cost * total_prompt_tokens + completion_cost * total_completion_tokens) / 1000

    return result, string_response, cost


def test_get_choices():
    print(get_choices(5))

def test_assign_context():
    choices = get_choices(2)
    pair = dict()
    pair["choices"] = choices
    print(assign_context(pair, "positive"))

def test_construct_choice_description():
    choice = [["black", "yellow"], ["brown", "blue"], ["red"]]
    print(construct_choice_description(choice, "positive"))

if __name__ == "__main__":
    # test_get_choices()
    # test_assign_context()
    # test_construct_choice_description()
    pass