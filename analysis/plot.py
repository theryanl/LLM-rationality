from datasets import load_dataset, load_from_disk
from rich import print
import json 
import matplotlib.pyplot as plt
# Load the dataset
import numpy as np
from scipy.stats import kstest, chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
file_path = "" # redacted for anonymity purposes
dataset = load_dataset('json', data_files='') # redacted for anonymity purposes


def moving_average(data, window_size):
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


def plot(ground_truth, pred, title):
    plt.figure(figsize=(10, 5))
    smoothed_ground_truth = moving_average(ground_truth, 10)
    smoothed_pred = moving_average(pred, 10)
    plt.plot(smoothed_pred, label='b_prob', marker='x', markersize=2, color='forestgreen')
    plt.plot(smoothed_ground_truth, label='b_rate', marker='o', markersize=2, color='gold')
    
    # plt.plot(ground_truth, label='b_rate', marker='o', markersize=2, color='gold')
    # plt.plot(pred, label='b_prob', marker='x', markersize=2, color='skyblue')
    plt.xlabel('num_instances')
    plt.ylabel('Probability')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(title.replace(" ", "_") + ".png")
    
    
    
if __name__ == "__main__":
    dataset = load_dataset('json', data_files='') # redacted for anonymity purposes
    dataset.save_to_disk('') # redacted for anonymity purposes
    loaded_dataset = load_from_disk('')['train'] # redacted for anonymity purposes
    
    print(f"Loaded dataset: {loaded_dataset}")
    b_rate = loaded_dataset['b_rate']
    count_B = 0
    count_A = 0
    ground_truth = []
    for rate in b_rate:
        if rate > 0.5:
            count_B += 1
            ground_truth.append("B")
        else:
            count_A += 1
            ground_truth.append("A")
    print(f"Count of B: {count_B / 5000}")
    print(f"Count of A: {count_A / 5000}")
    
    b_prob = []
    for bp in loaded_dataset['prob']:
        b_prob.append(bp['Prob_B'])
        
    a_rate = [1 - br for br in b_rate]
    a_prob = []
    for ap in loaded_dataset['prob']:
        a_prob.append(ap['Prob_A'])
    
    
    count_B = 0
    count_A = 0
    pred = []
    for prob in a_prob:
        if prob > 0.5:
            count_A += 1
            pred.append("A")
        else:
            count_B += 1
            pred.append("B")
    print(f"Count of B: {count_B / 5000}")
    print(f"Count of A: {count_A / 5000}")
    # Plot the data
    plot(a_rate, a_prob, "A_rate_vs_A_prob_GPT-4 choices choices13k")
    plot(b_rate, b_prob, "B_rate_vs_B_prob_GPT-4 choices choices13k")

    ks_stat, ks_pval = kstest(pred, ground_truth)
    observed = [[ground_truth.count('A'), ground_truth.count('B')],
            [pred.count('A'), pred.count('B')]]
    chi2_stat, p_val, dof, expected = chi2_contingency(observed)
    print(f"Chi-squared test statistic: {chi2_stat}")
    print(f"Chi-squared test p-value: {p_val}")
    print(f"Degrees of freedom: {dof}")
    print(f"Expected: {expected}")
    
    disagree_count = sum([1 for gt, pr in zip(ground_truth, pred) if gt != pr])
    table = [[sum((np.array(ground_truth) == 'A') & (np.array(pred) == 'A')), 
          sum((np.array(ground_truth) == 'A') & (np.array(pred) == 'B'))],
         [sum((np.array(ground_truth) == 'B') & (np.array(pred) == 'A')), 
          sum((np.array(ground_truth) == 'B') & (np.array(pred) == 'B'))]]
    result = mcnemar(table, exact=True)
    print(f"results.mcnemar: {result.statistic}")
    print(f"results.pvalue: {result.pvalue}")