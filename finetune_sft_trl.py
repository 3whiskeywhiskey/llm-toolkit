# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional
import torch

from datasets import load_dataset
from tqdm import tqdm
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    LlamaTokenizer,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig

from trl import SFTTrainer


tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl's SFTTrainer.
#
# This example fine-tunes any causal language model (GPT-2, GPT-Neo, etc.)
# by using the SFTTrainer from trl, we will leverage PEFT library to finetune
# adapters on the model.
#
########################################################################

@dataclass
class ScriptArguments:
    """
    Define the arguments used in this script.
    """

    model_name: Optional[str] = field(default="huggyllama/llama-7b", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="ybelkada/oasst1-tiny-subset", metadata={"help": "the dataset name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "Override name of output adapter"})
    use_8_bit: Optional[bool] = field(default=False, metadata={"help": "use 8 bit precision"})
    use_seq2seq_lm: Optional[bool] = field(default=False, metadata={"help": "use seq2seq LM"})
    use_4_bit: Optional[bool] = field(default=True, metadata={"help": "use 4 bit precision"})
    bnb_4bit_quant_type: Optional[str] = field(default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"})
    use_bnb_nested_quant: Optional[bool] = field(default=False, metadata={"help": "use nested quantization"})
    use_multi_gpu: Optional[bool] = field(default=False, metadata={"help": "use multi GPU"})
    use_adapters: Optional[bool] = field(default=True, metadata={"help": "use adapters"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "input batch size"})
    max_seq_length: Optional[int] = field(default=512, metadata={"help": "max sequence length"})
    optimizer_name: Optional[str] = field(default="adamw_hf", metadata={"help": "Optimizer name"})
    resume: Optional[str] = field(default=None, metadata={"help": "Resume from checkpoint"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "Maximum number of training steps"})

def get_current_device():
    return Accelerator().process_index

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.output_name is None:
    output_name = script_args.model_name.split('/')[-1] + '-' + script_args.dataset_name.split('/')[-1]
else:
    output_name = script_args.output_name

dataset = load_dataset(script_args.dataset_name, split="train[:10%]")

# We load the model
if script_args.use_multi_gpu:
    device_map = "auto"
else:
    device_map = {"":get_current_device()}

if script_args.use_8_bit and script_args.use_4_bit:
    raise ValueError(
        "You can't use 8 bit and 4 bit precision at the same time"
    )

if script_args.use_4_bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=script_args.use_bnb_nested_quant,
    )   
else:
    bnb_config = None

transformers_class = AutoModelForSeq2SeqLM if script_args.use_seq2seq_lm else AutoModelForCausalLM

model = transformers_class.from_pretrained(
    script_args.model_name, 
    load_in_8bit=script_args.use_8_bit, 
    load_in_4bit=script_args.use_4_bit,
    device_map=device_map if (script_args.use_8_bit or script_args.use_4_bit) else None,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

if script_args.use_adapters:
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM" if not script_args.use_seq2seq_lm else "SEQ_2_SEQ_LM",
    )
else:
    peft_config = None
    if script_args.use_8_bit:
        raise ValueError(
            "You need to use adapters to use 8 bit precision"
        )

model_names = ["llama", "guanaco"]
if any(name in script_args.model_name for name in model_names):
    tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
else:
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

training_arguments = TrainingArguments(
    per_device_train_batch_size=script_args.batch_size,
    max_steps=script_args.max_steps,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=script_args.batch_size,
    output_dir="./results", 
    report_to=["none"],
    optim=script_args.optimizer_name,
    fp16=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=peft_config,
    max_seq_length=script_args.max_seq_length,
    args=training_arguments,
)

trainer.train(resume_from_checkpoint=script_args.resume)

trainer.model.save_pretrained(f"./results/{output_name}")