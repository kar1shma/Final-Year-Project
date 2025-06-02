# #!/usr/bin/env bash

# # paths to GGUF models
# LLAMA2_PATH="models/llama-2-13b-chat.Q4_K_M.gguf"
# FLAN_PATH="models/flan-t5-base-q4_k_m.gguf"

# TASK_FILE="benchmark.json"

# models=("llama2" "mistral")
# paths=("$LLAMA2_PATH" "$MISTRAL_PATH")

# for i in "${!models[@]}"; do
#   model="${models[$i]}"
#   model_path="${paths[$i]}"
#   out="${model}_results.json"
#   echo "→ Starting ${model} (will write to ${out})"
#   python eval.py \
#     --model "$model" \
#     --model_path "$model_path" \
#     --tasks  "$TASK_FILE" \
#     --out    "$out"
#   echo "Finished ${model}"
#   echo
# done

# echo "All evaluations complete."


#!/usr/bin/env bash

# path to GGUF model
MISTRAL_PATH="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
TASK_FILE="mini_benchmark.json"
OUT_FILE="mistral_test_results.json"

echo "→ Starting Mistral (will write to ${OUT_FILE})"
python eval.py \
  --model mistral \
  --model_path "$MISTRAL_PATH" \
  --tasks "$TASK_FILE" \
  --out "$OUT_FILE"
echo "Finished Mistral"

echo "Evaluation complete."

