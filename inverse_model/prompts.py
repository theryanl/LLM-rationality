positive_instructions = """The following are two choices that people have made between different bags of candy. Each candy is a different color. 

Choice 1 {bag_choice_1}: 
{choice_1_bags}The person making the choice chose Bag {choice_1_chosen_bag}. 

Choice 2 {bag_choice_2}: 
{choice_2_bags}The person making the choice chose Bag {choice_2_chosen_bag}. 

People were required to choose among the bags available, and were not allowed to reject all the bags. For example, when there is only one bag, the person has no choice but to choose it. 
Which choice (1 or 2) more strongly suggests that the person making the choice likes {X} candies? {prompt_method}"""

bag_choice_single = "contained only one option"
bag_choice_multiple = "was made between the following bags"

positive_ranking_instructions = """
The following is a list of 47 choices that people have made between different bags of candy. Each candy is a different color. 

{choices_list}
People were required to choose among the bags available, and were not allowed to reject all the bags. For example, when there is only one bag, the person has no choice but to choose it. 

Your task is to sort these choices by how strongly they suggest that the person making the choice likes {X} candies. Please sort the choices from weakest evidence to strongest. Please include all the choices. 

When you are finished, write the choice numbers in the order you sorted them in, separated with a new line. If there are some choices that you think provide equal evidence, you can write their numbers on the same line. But even if you think there is only a small difference between two choices, please sort the choices into separate lines. The numbers were assigned to choices randomly, so they do not indicate anything about how the cards should be sorted. Please double-check the numbers you wrote down to make sure you didn't skip a choice or accidentally write down the wrong number. {prompt_method}
"""

choices_list = """Choice {index} was made between the following bags:
{choice_bags}The person making the choice chose Bag {chosen_bag}. 

"""

positive_context = ["black", "yellow", "brown", "blue", "red"]


negative_instructions = """A team of scientists is studying how people respond to painful electric shocks applied at different body locations. In the study, some participants are allowed to choose the set of shocks they receive. The following are two choices that study participants have made between different sets of electric shocks. 

Choice 1 {shock_choice_1}: 
{choice_1_sets}The person making the choice chose Set {choice_1_chosen_set}. 

Choice 2 {shock_choice_2}: 
{choice_2_sets}The person making the choice chose Set {choice_2_chosen_set}. 

Note that participants were required to choose among the sets available, and were not allowed to opt-out and reject all the sets. For example, when there is only one set of shocks in the choice, the person has no choice but to choose it. 
Which choice (1 or 2) more strongly suggests that the person making the choice finds shocks at {X} relatively tolerable? {prompt_method}"""

shock_choice_single = "contained only one option"
shock_choice_multiple = "was made between the following sets of shocks"

negative_context = ["shock location 1", "shock location 2", "shock location 3", "shock location 4", "shock location 5"]


cot_prompt = """Let's think step by step. """
# if we are concerned this is too expensive, we can add something like, "Explain your thoughts in 1 sentence"
# Structure your answer in the following manner: 
# Reasoning: [...]
# Answer: Choice [1 or 2]"""

zero_shot_prompt = """Please respond with either \"Choice 1\" or \"Choice 2\". Do not include anything else in your answer. """

re_query_prompt_positive = """Based on the following response, which choice (1 or 2) more strongly suggests that the person making the choice likes {X} candies? If the response indicates that choice 1 and choice 2 provide equally strong evidence, answer \"Tie\". But even if it indicates there is only a small difference between the two choices, please respond with the one that provides the stronger evidence: \"Choice 1\" or \"Choice 2\". Do not include anything else in your answer. 

Response: {prev_response}"""

re_query_prompt_negative = """Based on the following response, which choice (1 or 2) more strongly suggests that the person making the choice finds shocks at {X} relatively tolerable? If the response indicates that choice 1 and choice 2 provide equally strong evidence, answer \"Tie\". But even if it indicates there is only a small difference between the two choices, please respond with the one that provides the stronger evidence: \"Choice 1\" or \"Choice 2\". Do not include anything else in your answer. 

Response: {prev_response}"""