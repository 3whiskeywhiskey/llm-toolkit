# llm-toolkit
tools for training and inferencing LLMs including qlora adapters and quantized models

# Examples

Create a llama-7B openassistant-guanaco adapter

```
python3 finetune_sft_trl.py --use_multi_gpu True --model_name huggyllama/llama-7b --dataset_name timdettmers/openassistant-guanaco 
```