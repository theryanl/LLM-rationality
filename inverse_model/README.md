# Inverse Modeling Experiments
This directory contains all the contents for the inverse modeling task (making inferences about others' decisions). 

The original experiment materials are in: human_results_negative.txt, human_results_positive.txt, negative_weak_to_strong_fixed.txt, positive_weak_to_strong.txt.

Using these experiment materials, prompts are made in prompts.py.

Code to query the LLM results are in: inverse_model.py, fix_queries.py (for partial fixes). To run, use run_inverse.sh, run_inverse_neg.sh, or run_inverse_batch.sh. Batches are used for GPT-4o only in our implementation, but are recommended since they are 50% cheaper. 

Code to compute the human results are in: choicesort.m, choicesort_representativeness.m, printresults_fractional.py.

Code to run the analysis is in: analysis.py. To run, use run_analysis.sh. 
