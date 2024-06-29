# ./inverse_model/run_analysis.sh    
python3.11 -m inverse_model.analysis \
--total_completions 43 \
--prompt_method cot \
--prompt_type positive \
--debug True \
--model Llama-3-70B \
--construct_matrix_from_save_dict False \
--path inverse_model/results/llama/results_70B_pos_cot/ \
--file_prefix inverse_model \
# --path inverse_model/results/pairwise/{args.prompt_type}/{args.prompt_method}/" \