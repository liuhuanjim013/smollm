# SmolLM evaluation scripts

We're using the [LightEval](https://github.com/huggingface/lighteval/) library to benchmark our models. 

Check out the [quick tour](https://github.com/huggingface/lighteval/wiki/Quicktour) to configure it to your own hardware and tasks.

## Setup

Use conda/uv/venv with `python>=3.11`.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate
```

Adjust the pytorch installation according to your environment:
```bash
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
For reproducibility, we recommend fixed versions of the libraries:
```bash
uv pip install -r requirements.txt
```

## Running the evaluations

### SmolLM2 base models

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm \
  "pretrained=HuggingFaceTB/SmolLM2-1.7B,revision=main,dtype=bfloat16,gpu_memory_utilization=0.8,max_model_length=4096,data_parallel_size=4" \
  "smollm2_base.txt" --custom-tasks "tasks.py" --output-dir "./evals" --save-details
```

### SmolLM2 instruction-tuned models

(note the `--use_chat_template` flag)
```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm \
  "pretrained=HuggingFaceTB/SmolLM2-1.7B-Instruct,revision=main,dtype=bfloat16,gpu_memory_utilization=0.8,max_model_length=4096,data_parallel_size=4" \
  "smollm2_instruct.txt" --custom-tasks "tasks.py"  --use-chat-template --output-dir "./evals" --save-details
```

### FineMath dataset ablations

See the collection for model names: https://huggingface.co/collections/HuggingFaceTB/finemath-6763fb8f71b6439b653482c2

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm \
  "pretrained=HuggingFaceTB/finemath-ablation-4plus-160B,revision=main,dtype=bfloat16,gpu_memory_utilization=0.7,max_model_length=4096,data_parallel_size=4" \
  "custom|math|4|1,custom|gsm8k|5|1,custom|arc:challenge|0|1,custom|mmlu_pro|0|1,custom|hellaswag|0|1" --custom-tasks "tasks.py" --output-dir "./evals" --save-details
```
