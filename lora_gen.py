from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    HfArgumentParser
)
from peft import PeftModel    

@dataclass
class ScriptArguments:
    """
    Define the arguments used in this script.
    """

    model_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
    adapter: Optional[str] = field(default=None, metadata={"help": "the adapter (qlora) name"})
    use_4_bit: Optional[bool] = field(default=True, metadata={"help": "use 4 bit precision"})
    bnb_4bit_quant_type: Optional[str] = field(default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"})
    use_bnb_nested_quant: Optional[bool] = field(default=False, metadata={"help": "use nested quantization"})
    max_seq_length: Optional[int] = field(default=128, metadata={"help": "max sequence length"})
    optimizer_name: Optional[str] = field(default="adamw_hf", metadata={"help": "Optimizer name"})
    prompt: Optional[str] = field(default="How do you make butter?", metadata={"help": "The prompt for the AI assistant"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    load_in_4bit=script_args.use_4_bit,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory= {i: '16000MB' for i in range(torch.cuda.device_count())},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=script_args.use_4_bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=script_args.use_bnb_nested_quant,
        bnb_4bit_quant_type=script_args.bnb_4bit_quant_type
    ),
)
if script_args.adapter is not None:
    print("using adapter")
    model = PeftModel.from_pretrained(model, script_args.adapter)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

prompt = script_args.prompt
formatted_prompt = (
    f"A chat between a curious human and an artificial intelligence assistant."
    f"The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    f"### Human: {prompt} ### Assistant:"
)
inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda:0")
outputs = model.generate(inputs=inputs.input_ids, max_new_tokens=script_args.max_seq_length)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
