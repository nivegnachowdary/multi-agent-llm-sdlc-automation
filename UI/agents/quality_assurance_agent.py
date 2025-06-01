from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain_core.messages import AIMessage

MODEL_REPO = "Rahul-8799/quality_assurance_stablecode"

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    torch_dtype=torch.float16,
    device_map="auto"
)

def run(state: dict) -> dict:
    """Reviews code output and suggests improvements."""
    messages = state["messages"]
    prompt = messages[-1].content

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(input_ids, max_new_tokens=3000)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {
        "messages": [AIMessage(content=output)],
        "chat_log": state["chat_log"] + [{"role": "Quality Assurance", "content": output}],
        "qa_output": output,
    }