import torch
from transformers import pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen3-4B",
    model_kwargs={
        "quantization_config": quant_config,
        "device_map": "auto"
    },
    max_new_tokens=100,
    temperature=0.5
)

llm = HuggingFacePipeline(pipeline=pipe)

result = llm.invoke("Who is messi?")

print(result)
