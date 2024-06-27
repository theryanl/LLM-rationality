import json
import os 
import argparse
from tqdm import tqdm
from pathlib import Path
from random import random
from rich import print
import numpy as np
from src.data.data_utils import ChoicesDataset
from scipy.stats import pearsonr, spearmanr
from datasets import load_dataset, Dataset


def load_data(args):
    """
    Load the dataset
    """

    if args.dataset_name == "llama3-70b-individual-zero-shot":
        dataset = load_dataset("json", data_files="")
        # redacted for anonymity purposes
        dataset = dataset['train'].select(range(args.num_examples))
    return dataset

def cal_expected_value(prob_A, prob_B):
    """
    Calculate the expected value of A and B
    Parameters:
        prob_A: list [[prob, reward], [prob, reward], ...]
        prob_B: list [[prob, reward], [prob, reward], ...]
    """
    expected_A = 0
    expected_B = 0
    for prob, reward in prob_A:
        expected_A += round(prob, 4) * reward
    for prob, reward in prob_B:
        expected_B += round(prob, 4) * reward
    return expected_A, expected_B


def cal_pearson_corr(prob_pred, prob_expected, args):
    """
    Calculate the Pearson correlation between predicted and expected probabilities
    Parameters:
        prob_pred: list of predicted probabilities ("A" or "B")
        prob_expected: list of expected probabilities ("A" or "B")
    """
    if args.mode == "softmax":
        prob_pred_A = [p[0] for p in prob_pred]
        prob_expected_A = [p[0] for p in prob_expected]
        prob_pred_B = [p[1] for p in prob_pred]
        prob_expected_B = [p[1] for p in prob_expected]
        pearsonr_A, preason_p_value_A = pearsonr(prob_pred_A, prob_expected_A)
        pearsonr_B, preason_p_value_B = pearsonr(prob_pred_B, prob_expected_B)
        return pearsonr_A, preason_p_value_A, pearsonr_B, preason_p_value_B
    elif args.mode == "max":
        prob_pred = [1 if p == "A" else 0 for p in prob_pred]
        # print(f"Prob pred: {prob_pred}")
        prob_expected = [1 if p == "A" else 0 for p in prob_expected]
        # print(f"Prob expected: {prob_expected}")
        return pearsonr(prob_pred, prob_expected)


def cal_spearman_corr(prob_pred, prob_expected, args):
    """
    Calculate the Spearman correlation
    Parameters:
        prob_pred: list of predicted probabilities ("A" or "B")
        prob_expected: list of expected probabilities ("A" or "B")
    """
    if args.mode == "softmax":
        prob_pred_A = [p[0] for p in prob_pred]
        prob_expected_A = [p[0] for p in prob_expected]
        prob_pred_B = [p[1] for p in prob_pred]
        prob_expected_B = [p[1] for p in prob_expected]
        spearmanr_A, spearman_p_value_A = spearmanr(prob_pred_A, prob_expected_A)
        spearmanr_B, spearman_p_value_B = spearmanr(prob_pred_B, prob_expected_B)
        return spearmanr_A, spearman_p_value_A, spearmanr_B, spearman_p_value_B
    elif args.mode == "max":
        prob_pred = [1 if p == "A" else 0 for p in prob_pred]
        prob_expected = [1 if p == "A" else 0 for p in prob_expected]
        return spearmanr(prob_pred, prob_expected)
    

def cal_mean_squared_error(prob_pred, prob_expected, args):
    """
    Calculate the mean squared error between predicted and expected probabilities.
    
    Parameters:
        prob_pred: list of human choices ('A' or 'B')
        prob_expected: list of expected choices ('A' or 'B')
    
    Returns:
        Mean squared error
    """
    if args.mode == "softmax":
        # Converting 'A' to 1 and 'B' to 0
        num_pred_A = [p[0] for p in prob_pred]
        num_expected_A = [p[0] for p in prob_expected]
        num_pred_B = [p[1] for p in prob_pred]
        num_expected_B = [p[1] for p in prob_expected]
        
        # Calculate the MSE
        mse_A = sum((p - e) ** 2 for p, e in zip(num_pred_A, num_expected_A)) / len(num_pred_A)
        mse_B = sum((p - e) ** 2 for p, e in zip(num_pred_B, num_expected_B)) / len(num_pred_B)
        return mse_A, mse_B
    elif args.mode == "max":
        # Converting 'A' to 1 and 'B' to 0
        num_pred = [1 if x == 'A' else 0 for x in prob_pred]
        num_expected = [1 if x == 'A' else 0 for x in prob_expected]
        
        # Calculate the MSE
        mse = sum((p - e) ** 2 for p, e in zip(num_pred, num_expected)) / len(num_pred)
        return mse


def main(args):
    
    model_dataset_zero_shot = load_dataset("json", data_files="") # redacted for anonymity purposes
    model_dataset_zero_shot = model_dataset_zero_shot['train'].select(range(args.num_examples))
    print(f"Model dataset zero shot: {model_dataset_zero_shot}")
    model_dataset_cot = load_dataset("json", data_files="") # redacted for anonymity purposes
    model_dataset_cot = model_dataset_cot['train'].select(range(args.num_examples))
    print(f"Model dataset cot: {model_dataset_cot}")
    
    dataset = ChoicesDataset(
        begin_idx = args.begin_idx,
        num_samples=args.num_examples,
        seed=args.seed,
        instruction_mode=args.instruction_mode,
    )
    human_samples = dataset.samples
    samples = dataset.reward_samples
   
    expected_choices = []
    human_choices = []  
    model_zero_shot_choices = []
    model_cot_choices = []
    output_file = f"" # redacted for anonymity purposes
    
   
    for sample, human_sample, model_zero_pred, model_cot_pred in zip(samples, human_samples, model_dataset_zero_shot, model_dataset_cot):
        expected_A, expected_B = cal_expected_value(sample.prob_A, sample.prob_B)
        expected_choices.append("A" if expected_A > expected_B else "B")
        human_choices.append("B" if human_sample.b_rate >= 0.5 else "A")
       
        if args.n_completions == "n":
            model_zero_shot_choices.append("A" if model_zero_pred['prob']["Prob_A"] > model_zero_pred['prob']["Prob_B"] else "B")
        else:
            model_zero_shot_choices.append(model_zero_pred['answer'][0])
            
        model_cot_choices.append(model_cot_pred['answer'][0])
        
        
    # model_zero_expected_spearman, model_zero_expected_spearman_p_value = cal_spearman_corr(model_zero_shot_choices, expected_choices, args)
    # model_zero_expected_pearson, model_zero_expected_pearson_p_value = cal_pearson_corr(model_zero_shot_choices, expected_choices, args)
    # model_zero_expected_mse = cal_mean_squared_error(model_zero_shot_choices, expected_choices, args)
    # print(f"Model zero shot expected Spearman correlation: {model_zero_expected_spearman} | Model zero shot expected Pearson correlation: {model_zero_expected_pearson} | Model zero shot expected MSE: {model_zero_expected_mse}")
    
    # model_cot_expected_spearman, model_cot_expected_spearman_p_value = cal_spearman_corr(model_cot_choices, expected_choices, args)
    # model_cot_expected_pearson, model_cot_expected_pearson_p_value = cal_pearson_corr(model_cot_choices, expected_choices, args)
    # model_cot_expected_mse = cal_mean_squared_error(model_cot_choices, expected_choices, args)
    # print(f"Model cot expected Spearman correlation: {model_cot_expected_spearman} | Model cot expected Pearson correlation: {model_cot_expected_pearson} | Model cot expected MSE: {model_cot_expected_mse}")
    
    # model_zero_human_spearman, model_zero_human_spearman_p_value = cal_spearman_corr(model_zero_shot_choices, human_choices, args)
    # model_zero_human_pearson, model_zero_human_pearson_p_value = cal_pearson_corr(model_zero_shot_choices, human_choices, args)
    # model_zero_human_mse = cal_mean_squared_error(model_zero_shot_choices, human_choices, args)
    # print(f"Model zero shot human Spearman correlation: {model_zero_human_spearman} | Model zero shot human Pearson correlation: {model_zero_human_pearson} | Model zero shot human MSE: {model_zero_human_mse}")   
    
    # model_cot_human_spearman, model_cot_human_spearman_p_value = cal_spearman_corr(model_cot_choices, human_choices, args)
    # model_cot_human_pearson, model_cot_human_pearson_p_value = cal_pearson_corr(model_cot_choices, human_choices, args)
    # model_cot_human_mse = cal_mean_squared_error(model_cot_choices, human_choices, args)
    # print(f"Model cot human Spearman correlation: {model_cot_human_spearman} | Model cot human Pearson correlation: {model_cot_human_pearson} | Model cot human MSE: {model_cot_human_mse}")
            
    
    # human_expected_spearman, human_expected_spearman_p_value = cal_spearman_corr(human_choices, expected_choices, args)
    # human_expected_pearson, human_expected_pearson_p_value = cal_pearson_corr(human_choices, expected_choices, args)
    # human_expected_mse = cal_mean_squared_error(human_choices, expected_choices, args)
    # print(f"Human expected Spearman correlation: {human_expected_spearman} | Human expected Pearson correlation: {human_expected_pearson} | Human expected MSE: {human_expected_mse}")
        
    
    model_zero_model_cot_spearman, model_zero_model_cot_spearman_p_value = cal_spearman_corr(model_zero_shot_choices, model_cot_choices, args)
    model_zero_model_cot_pearson, model_zero_model_cot_pearson_p_value = cal_pearson_corr(model_zero_shot_choices, model_cot_choices, args)
    model_zero_model_cot_mse = cal_mean_squared_error(model_zero_shot_choices, model_cot_choices, args)
    print(f"Model zero shot model cot Spearman correlation: {model_zero_model_cot_spearman} | Model zero shot model cot Pearson correlation: {model_zero_model_cot_pearson} | Model zero shot model cot MSE: {model_zero_model_cot_mse}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin_idx", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="max")
    parser.add_argument("--subset", type=bool, default=True)
    parser.add_argument("--dataset_name", type=str, default="gpt-4-aggregate-cot")
    parser.add_argument("--instruction_mode", type=str, default="aggregate_cot")
    parser.add_argument("--n_completions", type=str, default="1")
    args = parser.parse_args()
    main(args)
    
    print(f"[bold green]Finished forward modeling analysis[/bold green]")
    