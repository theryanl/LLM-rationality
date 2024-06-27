"""
python -m data.data_utils
"""
from datasets import load_dataset, Dataset
from rich import print
import pandas as pd
import random
import contextlib
from tqdm import tqdm
from typing import List, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np


class PromptConstants:
    INDIVIDUAL_ZERO_SHOT = """A person is presented with two gambling machines, and makes a choice between the machines with the goal of maximizing the amount of dollars received. The person will get one reward from the machine they choose. A fixed proportion of 10% of this value will be paid to the participant as a performance bonus. If the reward is negative, their bonus is set to $0. \n\nMachine A: {} \nMachine B: {} \n\nWhich machine do you choose? \nDo not provide any explanation, only answer with A or B: """
    INDIVIDUAL_COT = """A person is presented with two gambling machines, and makes a choice between the machines with the goal of maximizing the amount of dollars received. The person will get one reward from the machine they choose. A fixed proportion of 10% of this value will be paid to the participant as a performance bonus. If the reward is negative, their bonus is set to $0. \n\nMachine A: {} \nMachine B: {} \n\nWhich machine does the person choose? \nLet's think step by step before providing the final output. Please provide the human choice in the json format with the key "choice" and value "A" or "B"."""
    ACT_INDIVIDUAL_ZERO_SHOT = """There are two gambling machines, A and B. You need to make a choice between the machines with the goal of maximizing the amount of dollars received. You will get one reward from the machine that you choose. A fixed proportion of 10% of this value will be paid to you as a performance bonus. If the reward is negative, your bonus is set to $0. \n\nMachine A: {} \nMachine B: {} \n\nWhich machine do you choose? \nDo not provide any explanation, only answer with A or B: """
    ACT_INDIVIDUAL_COT = """There are two gambling machines, A and B. You need to make a choice between the machines with the goal of maximizing the amount of dollars received. You will get one reward from the machine that you choose. A fixed proportion of 10% of this value will be paid to you as a performance bonus. If the reward is negative, your bonus is set to $0. \n\nMachine A: {} \nMachine B: {} \n\nWhich machine do you choose? \nLet's think step by step before answering with A or B. Please provide the your choice in the json format with the key "choice" and value "A" or "B" """
    AGGREGATE_ZERO_SHOT = """{} people are presented with two gambling machines, and each person makes a choice between the machines with the goal of maximizing the amount of dollars received. Each person will get one reward from the machine they choose. A fixed proportion of 10% of this value will be paid to the participant as a performance bonus. If the reward is negative, their bonus is set to $0. \n\nMachine A: {} \nMachine B: {} \n\nHow many people choose Machine A? \nHow many people choose Machine B? \n\nPlease only provide the precentage of people who choose Machine A and Machine B in the json format."""
    AGGREGATE_COT = """{} people are presented with two gambling machines, and each person makes a choice between the machines with the goal of maximizing the amount of dollars received. Each person will get one reward from the machine they choose. A fixed proportion of 10% of this value will be paid to the participant as a performance bonus. If the reward is negative, their bonus is set to $0. \n\nMachine A: {} \nMachine B: {} \n\nHow many people choose Machine A? \nHow many people choose Machine B? \n\nLet's think step by step before providing the final output. Please provide the precentage of people who choose Machine A and Machine B in the json format."""


@dataclass
class Sample:
    sample_id: int = None
    input_text: str = None
    num_p: int = None
    b_rate: float = None
    b_rate_std: float = None
    def to_dict(self):
        return asdict(self)
    

@dataclass
class RewardSample:
    sample_id: int = None
    prob_A: list = None
    prob_B: list = None
    

@contextlib.contextmanager
def temp_seed(seed):
    np_state = np.random.get_state()
    python_random_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(python_random_state)
        
        
        
class ChoicesDataset:
    def __init__(
        self,
        begin_idx=None,
        num_samples=None,
        seed=None,
        instruction_mode=None,
    ):
        self.num_samples = num_samples
        self.begin_idx = begin_idx
        self.seed = seed
        self.instruction_mode = instruction_mode
        ds, c13k_problems = self.load_data()
        ds = ds.select(range(begin_idx, begin_idx + num_samples))
        c13k_problems = c13k_problems.select(range(begin_idx, begin_idx + num_samples))
        print(f"ds: {ds}")
        print(f"c13k_problems: {c13k_problems}")
        assert len(ds) == len(c13k_problems)
        self.prob_samples = [self.build_reward_sample(c13k_problems[i]) for i in range(len(c13k_problems))]
        self.prompt_samples = self.get_samples(ds, c13k_problems)
        self.config = dict(
            begin_idx = begin_idx,
            num_samples=len(self.prompt_samples),
            seed=seed,
        )
        
    @property
    def samples(self):
        return self.prompt_samples
    
    @property
    def reward_samples(self):
        return self.prob_samples
        
    def load_data(self):
        # load dataset from csv file
        dataset = load_dataset("")["train"] # redacted for anonymity purposes
        c13k_problems = pd.read_json("", orient='index') # redacted for anonymity purposes
        c13k_problems = Dataset.from_pandas(c13k_problems)
        return dataset, c13k_problems
    
    def build_reward_sample(self, example):
        sample_id = example["__index_level_0__"]
        prob_A = example["A"]
        prob_B = example["B"]
        reward_sample = RewardSample(sample_id=sample_id, prob_A=prob_A, prob_B=prob_B)
        return reward_sample
    
    def build_sample(self, example_ds, example_13k):
        sample_id = example_13k["__index_level_0__"]
        num_p = example_ds['n']

        # shuffle the order of the rewards 
        # #(We don't shuffle A and B together because the order of A and B should be the same for the same sample.)
        example_A = example_13k['A']
        random.shuffle(example_A)
        example_B = example_13k['B']
        random.shuffle(example_B)
        
        machine_A = ""  
        for i, (prob, outcome) in enumerate(example_A):
            prob = prob * 100
            prob = round(prob, 2)
            machine_A += f"${outcome} with {prob}% chance"
            if i == len(example_13k["A"]) - 1:
                machine_A += "."
            else:
                machine_A += ", "
        machine_B = ""
        for i, (prob, outcome) in enumerate(example_B):
            prob = prob * 100
            prob = round(prob, 2)
            machine_B += f"${outcome} with {prob}% chance"
            if i == len(example_13k["B"]) - 1:
                machine_B += "."
            else:
                machine_B += ", "

        if self.instruction_mode == "individual_zero_shot":
            instruction = PromptConstants.INDIVIDUAL_ZERO_SHOT
            input_text = instruction.format(
                    machine_A, machine_B
                )
        elif self.instruction_mode == "individual_cot":
            instruction = PromptConstants.INDIVIDUAL_COT
            input_text = instruction.format(
                    machine_A, machine_B
                )
        elif self.instruction_mode == "aggregate_zero_shot":
            instruction = PromptConstants.AGGREGATE_ZERO_SHOT
            input_text = instruction.format(
                    num_p, machine_A, machine_B
                )
        elif self.instruction_mode == "aggregate_cot":
            instruction = PromptConstants.AGGREGATE_COT
            input_text = instruction.format(
                    num_p, machine_A, machine_B
                )
        elif self.instruction_mode == "act_individual_zero_shot":
            instruction = PromptConstants.ACT_INDIVIDUAL_ZERO_SHOT
            input_text = instruction.format(
                    machine_A, machine_B
                )
        elif self.instruction_mode == "act_individual_cot":
            instruction = PromptConstants.ACT_INDIVIDUAL_COT
            input_text = instruction.format(
                    machine_A, machine_B
                )
        else:
            raise ValueError(f"Invalid instruction_mode: {self.instruction_mode}")

        num_p = example_ds['n']
        b_rate = example_ds['bRate']
        b_rate_std = example_ds['bRate_std']
    
        sample = Sample(sample_id=sample_id, 
                        input_text=input_text, 
                        num_p=num_p, 
                        b_rate=b_rate, 
                        b_rate_std=b_rate_std)
        return sample
    
    def get_samples(self, ds, c13k_problems):
        samples = [self.build_sample(ds[i], c13k_problems[i]) for i in range(len(ds))]
        return samples
        
        
if __name__ == "__main__":
    num_samples = 10
    seed = 42
    ds = ChoicesDataset(num_samples=num_samples, seed=seed)
    samples = ds.samples
    print(f"samples: {samples}")
    
    