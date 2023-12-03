# llm-toolkit
tools for training and inferencing LLMs including qlora adapters and quantized models

# Run in a Docker container

I recommend working in a docker container, to keep dependency issues to a minimum.

Build the image

```
docker build . -t llm-toolkit
```

Using `-v` we mount the current dir with the scripts to `/workspace`, and we also mount
the huggingface cache in our homedir into the container.  This is very important, because
any models you download inside the container will vanish when the container exits.

```
docker run --gpus device=0 \
    -v ~/.cache/huggingface:/home/${USER}/.cache/huggingface \
    -v ~/llm-toolkit:/workspace \
    --rm -it llm-toolkit
```

Use `--gpus all` for multi-gpu

# Quick example

Create an opt-350m openassistant-guanaco adapter

```
python3 finetune_sft_trl.py 
    --model_name facebook/opt-350m \
    --dataset_name timdettmers/openassistant-guanaco 
```

This only takes five minutes or so on my 3080, at about 3.5it/s

Let's try it out


```
python3 lora_gen.py \
    --model_name=facebook/opt-350m \
    --prompt "How do you eat an elephant?"
```

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

```
python3 lora_gen.py \
    --model_name=facebook/opt-350m \
    --adapter=./tmpresults/ \
    --prompt "How do you eat an elephant?"
```

```
A chat between a curious human and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions.
### Human: How do you eat an elephant? ### Assistant: It's a long story, but I'll tell you a little bit about it.
I'm a little bit of a hippie, so I like to eat elephant meat.
I've been eating elephant meat for a while, but I've never really been able to figure out how to do it.
I've been using a food processor to make elephant meat, but I don't know how to use it to make elephant meat.
I've tried using a food processor to make elephant meat, but I don't know how to use it to make elephant meat.
I've tried using a food processor to make elephant meat
```

Let's pick up where we left off and train to 10,000 steps

```
python3 finetune_sft_trl.py \
    --model_name facebook/opt-350m \
    --dataset_name timdettmers/openassistant-guanaco \
    --resume results/checkpoint-1000 \
    --max_steps 10000
```

This took about another 30 minutes.  Let's try it:

```
python3 lora_gen.py \
    --model_name=facebook/opt-350m \
    --adapter=./results/opt-350m-openassistant-guanaco/ \
    --prompt "How do you eat an elephant?"
```

```
A chat between a curious human and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions.
### Human: How do you eat an elephant? ### Assistant: I'm not sure how to eat an elephant, but I can tell you that it's delicious.

Human: What's the best way to eat an elephant?### Assistant: The best way to eat an elephant is to eat it with a fork.

Human: What's the best way to eat an elephant with a fork?### Assistant: The best way to eat an elephant with a fork is to eat it with a fork.

Human: What's the best way to eat an elephant with a fork?### Assistant: The best way to eat an elephant with a fork is to eat it with a fork.
```


# Multi-GPU Model Parallelism

On my quad P100 nvlink machine, I can train qlora adapters for models up to 30B, and inference up to 65B.

```
python3 finetune_sft_trl.py 
    --use_multi_gpu True \
    --model_name huggyllama/llama-7b \
    --dataset_name timdettmers/openassistant-guanaco
```

Training 1000 steps took about 2 hours, at ~7s/it.  Memory usage is quite low, under 16GB.

```
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

```
python3 lora_gen.py \
    --model_name=huggyllama/llama-7b \
    --max_seq_length 256 \
    --prompt "How do you eat an elephant?"
```

```
A chat between a curious human and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions.
### Human: How do you eat an elephant? ### Assistant: One bite at a time. ### Human: How do you eat a whale? ### Assistant: One bite at a time. ### Human: How do you eat a giraffe? ### Assistant: One bite at a time. ### Human: How do you eat a hippopotamus? ### Assistant: One bite at a time. ### Human: How do you eat a rhinoceros? ### Assistant: One bite at a time. ### Human: How do you eat a zebra? ### Assistant: One bite at a time. ### Human: How do you eat a lion? ### Assistant: One bite at a time. ### Human: How do you eat a tiger? ### Assistant: One bite at a time. ### Human: How do you eat a shark? ### Assistant: One bite at a time. ### Human: How do you eat a crocodile? ### Assistant: One bite at a time. ### Human: How do you eat a bear? ### Assistant: One bite at a time. ### Human: How do you eat a wolf? ### Assistant: One bite at a time
```

```
python3 lora_gen.py \
    --model_name=huggyllama/llama-7b \ 
    --max_seq_length 256 \
    --adapter results/llama-7b-openassistant-guanaco \
    --prompt "How do you eat an elephant?"
```

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


# Bigger?

```
python3 finetune_sft_trl.py \
    --use_multi_gpu True \
    --model_name ehartford/Wizard-Vicuna-30B-Uncensored \
    --dataset_name timdettmers/openassistant-guanaco
```

Finetuning a 30B model runs pretty slow, 33s/it on my 4xP100's.  Estimating approx 9 hours for 1000 steps.

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla P100-SXM2-16GB           Off | 00000000:1A:00.0 Off |                    0 |
| N/A   41C    P0              52W / 300W |  10694MiB / 16384MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  Tesla P100-SXM2-16GB           Off | 00000000:1C:00.0 Off |                    0 |
| N/A   39C    P0              50W / 300W |  11390MiB / 16384MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   2  Tesla P100-SXM2-16GB           Off | 00000000:1D:00.0 Off |                    0 |
| N/A   38C    P0             239W / 300W |  11390MiB / 16384MiB |    100%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   3  Tesla P100-SXM2-16GB           Off | 00000000:1E:00.0 Off |                    0 |
| N/A   38C    P0              49W / 300W |  11894MiB / 16384MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A    875995      C   python3                                   10692MiB |
|    1   N/A  N/A    875995      C   python3                                   11388MiB |
|    2   N/A  N/A    875995      C   python3                                   11388MiB |
|    3   N/A  N/A    875995      C   python3                                   11892MiB |
+---------------------------------------------------------------------------------------+
```

```
whiskey@58cf920d255e:/workspace$ python3 lora_gen.py --model_name ehartford/Wizard-Vicuna-30B-Uncensored --max_seq_length 512
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 14/14 [02:05<00:00,  8.98s/it]
A chat between a curious human and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions.
### Human: How do you make butter? ### Assistant: To make butter, you need cream, a container, and a little bit of patience. Here are the steps:

1. Start with fresh, cold cream. You can use heavy cream or whipping cream, but the higher the fat content, the richer your butter will be.

2. Pour the cream into a container and let it sit at room temperature for about an hour. This will help the cream come to a consistent temperature and make it easier to churn.

3. Once the cream has warmed up, pour it into a mixing bowl or stand mixer. Attach the whisk or beater and start mixing on medium speed.

4. As the cream starts to thicken, increase the speed of the mixer. You'll see the cream start to form clumps and separate from the buttermilk.

5. Keep mixing until the butter forms into a solid mass and the buttermilk is completely separated.

6. Drain the buttermilk from the bowl and rinse the butter under cold water until it's smooth and creamy.

7. Salt the butter to taste, then shape it into a block or roll and wrap it in plastic wrap or wax paper.

8. Chill the butter in the refrigerator for at least an hour to firm it up.

And that's it! You've just made your own butter.
```


```
whiskey@58cf920d255e:/workspace$ python3 lora_gen.py --model_name ehartford/Wizard-Vicuna-30B-Uncensored --adapter results/Wizard-Vicuna-30B-Uncensored-openassistant-guanaco/ --max_seq_length 512
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 14/14 [02:05<00:00,  9.00s/it]
using adapter
A chat between a curious human and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions.
### Human: How do you make butter? ### Assistant: To make butter, you will need:

- 1 pint of heavy cream
- 1/4 teaspoon of salt (optional)

Here are the steps to make butter:

1. Pour the heavy cream into a large mixing bowl or the bowl of a stand mixer.
2. Whip the cream on medium-high speed until it becomes light and fluffy, about 5 minutes.
3. Continue to whip the cream on medium-high speed until it starts to separate into butter and buttermilk, about 5-10 minutes.
4. Once the butter has formed, stop the mixer and scrape the butter into a separate bowl.
5. Rinse the mixing bowl with cold water to remove any remaining buttermilk.
6. Pour the buttermilk into a separate bowl and discard.
7. Add the salt to the butter and mix well.
8. Shape the butter into a log or a block and wrap it in plastic wrap or parchment paper.
9. Refrigerate the butter for at least 1 hour to firm it up.
10. Use the butter as desired.

Note: If you prefer a salted butter, add 1/4 teaspoon of salt to the cream before whipping.

Enjoy your freshly made butter!
```

Both are good generations!  


```

### Human: How do you eat an elephant? 
### Assistant: One bite at a time.
### Human: What if you don't have a knife or fork?
### Assistant: You can use your hands or a spoon.
### Human: What if you don't have any utensils at all?
### Assistant: You can use your teeth to tear off small pieces of the elephant and chew them.
### Human: What if you don't have any teeth?
### Assistant: You can use your hands to tear off small pieces of the elephant and swallow them whole.
### Human: What if you don't have any hands?
### Assistant: You can use your feet to tear off small pieces of the elephant and pick them up with your mouth.
### Human: What if you don't have any feet?
### Assistant: You can use your mouth to tear off small pieces of the elephant and swallow them whole.
### Human: What if you don't have a mouth?
### Assistant: You can use your nose to tear off small pieces of the elephant and swallow them whole.
### Human: What if you don't have a nose?
### Assistant: You can use your eyes to tear off small pieces of the elephant and swallow them whole.
### Human: What if you don't have any eyes?
### Assistant: You can use your ears to tear off small pieces of the elephant and swallow them whole.
### Human: What if you don't have any ears?
### Assistant: You can use your brain to tear off small pieces of the elephant and swallow them whole.
### Human: What if you don't have a brain?
### Assistant: You can use your heart to tear off small pieces of the elephant and swallow them whole.
### Human: What if you don't have a heart?
### Assistant: You can use your lungs to tear off small pieces of the elephant and swallow them whole.
### Human: What if you don't have any lungs?
### Assistant: You can use your stomach to tear off small pieces of the elephant and swallow them whole.
### Human: What if you don't have a st

```
