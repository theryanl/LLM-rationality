python3.11 -m src.inverse_model.inverse_model \
--output_dir src/inverse_model/results/pairwise/negative/zero_shot/ \
--num_choices 47 \
--prompt_type negative \
--prompt_method zero_shot \
--num_completions 42 \
--experiment_type pairwise \
--c_index_start 19 \
--split 2 \
--num_splits 1 \
# --load_previous src/inverse_model/results/pairwise/positive/zero_shot/
