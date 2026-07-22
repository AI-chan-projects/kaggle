import llm_20_questions
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import warnings

# FutureWarning 무시 설정
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="bitsandbytes"
)

llm = llm_20_questions
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
    )

# gemma-1.1-2b-it from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-1.1-2b-it",
    dtype=torch.bfloat16,
    quantization_config=quantization_config,
    device_map="auto"
)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("mps")

outputs = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
