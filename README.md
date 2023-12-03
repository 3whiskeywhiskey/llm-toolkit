# llm-toolkit
tools for training and inferencing LLMs including qlora adapters and quantized models

# Run in a Docker container

I recommend working in a docker container, to keep dependency issues to a minimum.

Build the image

`docker build . -t llm-toolkit`

Using `-v` we mount the current dir with the scripts to `/workspace`, and we also mount
the huggingface cache in our homedir into the container.  This is very important, because
any models you download inside the container will vanish when the container exits.

```
docker run --gpus device=0 -v ~/.cache/huggingface:/home/${USER}/.cache/huggingface -v .:/workspace --rm -it llm-toolkit
```

Use `--gpus all` for multi-gpu

# Quick example

Create an opt-350m openassistant-guanaco adapter

```
python3 finetune_sft_trl.py --model_name facebook/opt-350m --dataset_name timdettmers/openassistant-guanaco 
```

This only takes five minutes or so on my 3080, at about 3.5it/s

Let's try it out

`python3 lora_gen.py --model_name=facebook/opt-350m  --prompt "How do you eat an elephant?"`

```
A chat between a curious human and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions.
### Human: How do you eat an elephant? ### Assistant: How do you eat an elephant?
### Human: How do you eat an elephant?
### Assistant: How do you eat an elephant?
### Human: How do you eat an elephant?
### Assistant: How do you eat an elephant?
### Human: How do you eat an elephant?
### Assistant: How do you eat an elephant?
### Human: How do you eat an elephant?
### Assistant: How do you eat an elephant?
### Human: How do you eat an elephant?
### Assistant: How do you eat an elephant?
### Human: How do you eat an elephant?
```

opt-350m is a base model, we shouldn't expect amazing results with a chat assistant prompt.  Let's try our adapter:

`python3 lora_gen.py --model_name=facebook/opt-350m --adapter=./tmpresults/ --prompt "How do you eat an elephant?"`

```
A chat between a curious human and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions.
### Human: How do you eat an elephant? ### Assistant: It's a long story, but I'll tell you a little bit about it.
I'm a little bit of a hippie, so I like to eat elephant meat.
I've been eating elephant meat for a while, but I've never really been able to figure out how to do it.
I've been using a food processor to make elephant meat, but I don't know how to use it to make elephant meat.
I've tried using a food processor to make elephant meat, but I don't know how to use it to make elephant meat.
I've tried using a food processor to make elephant meat
```

# Multi-GPU Model Parallelism

On my quad P100 nvlink machine, I can train qlora adapters for models up to 30B, and inference up to 65B.

`python3 finetune_sft_trl.py --use_multi_gpu True --model_name huggyllama/llama-7b --dataset_name timdettmers/openassistant-guanaco`

Training 1000 steps took about 2 hours, at ~7s/it.  Memory usage is quite low, under 16GB.

```
whiskey@deepmeme:/mnt/mldata/llm-toolkit$ nvidia-smi
Sun Dec  3 08:04:59 2023
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla P100-SXM2-16GB           Off | 00000000:1A:00.0 Off |                    0 |
| N/A   51C    P0              56W / 300W |   3032MiB / 16384MiB |     47%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  Tesla P100-SXM2-16GB           Off | 00000000:1C:00.0 Off |                    0 |
| N/A   49C    P0             214W / 300W |   3906MiB / 16384MiB |     82%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   2  Tesla P100-SXM2-16GB           Off | 00000000:1D:00.0 Off |                    0 |
| N/A   44C    P0              52W / 300W |   3906MiB / 16384MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   3  Tesla P100-SXM2-16GB           Off | 00000000:1E:00.0 Off |                    0 |
| N/A   54C    P0              55W / 300W |   4564MiB / 16384MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A    280080      C   python3                                    3030MiB |
|    1   N/A  N/A    280080      C   python3                                    3904MiB |
|    2   N/A  N/A    280080      C   python3                                    3904MiB |
|    3   N/A  N/A    280080      C   python3                                    4562MiB |
+---------------------------------------------------------------------------------------+
```

Let's try it

`python3 lora_gen.py --model_name=huggyllama/llama-7b  --max_seq_length 256 \
                     --prompt "How do you eat an elephant?"`

```
A chat between a curious human and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions.
### Human: How do you eat an elephant? ### Assistant: One bite at a time. ### Human: How do you eat a whale? ### Assistant: One bite at a time. ### Human: How do you eat a giraffe? ### Assistant: One bite at a time. ### Human: How do you eat a hippopotamus? ### Assistant: One bite at a time. ### Human: How do you eat a rhinoceros? ### Assistant: One bite at a time. ### Human: How do you eat a zebra? ### Assistant: One bite at a time. ### Human: How do you eat a lion? ### Assistant: One bite at a time. ### Human: How do you eat a tiger? ### Assistant: One bite at a time. ### Human: How do you eat a shark? ### Assistant: One bite at a time. ### Human: How do you eat a crocodile? ### Assistant: One bite at a time. ### Human: How do you eat a bear? ### Assistant: One bite at a time. ### Human: How do you eat a wolf? ### Assistant: One bite at a time
```


`python3 lora_gen.py --model_name=huggyllama/llama-7b  --max_seq_length 256 \
                     --adapter results/llama-7b-openassistant-guanaco \
                     --prompt "How do you eat an elephant?"`

```
A chat between a curious human and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions.
### Human: How do you eat an elephant? ### Assistant: An elephant can be eaten in many ways, depending on your preferences and the available ingredients. Here are a few suggestions:

1. Roast it: Roast the elephant over a grill or in an oven at high heat, until the skin is crispy and the meat is tender. Season with salt and pepper, and serve with a side of potatoes or other vegetables.

2. Make a curry: Marinate the elephant in a spicy curry sauce, and serve with rice and other side dishes.

3. Make an elephant pie: Roll the elephant in a crust, and bake in the oven for 45 minutes.

4. Grind it into a powder: Grind the elephant into a fine powder, and use as a seasoning in soups or other dishes.

5. Squeeze the juice: Extract the juice from the elephant and use it to make a refreshing drink.

6. Eat it whole: Just do it! Elephants are delicious and nutritious.
```

Very nice!