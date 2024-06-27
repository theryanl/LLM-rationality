python3.11 -m src.inverse_model.inverse_model \
--output_dir src/inverse_model/batches/gpt-4o/negative/cot/ \
--prompt_type negative \
--prompt_method cot \
--num_completions 42 \
--experiment_type pairwise \
--c_index_start 0 \
--model gpt-4o-2024-05-13 \
--call_method batch \
--batch_purpose results \