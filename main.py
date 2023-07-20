from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = FastAPI()

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

class Item(BaseModel):
    prompt: str
    max_length: int = 50

@app.post("/generate")
def generate(item: Item):
    # Tokenize the input prompt
    input_ids = tokenizer.encode(item.prompt, return_tensors="pt")
    
    # Generate text
    output = model.generate(input_ids, max_length=item.max_length, num_return_sequences=1, do_sample=True)
    
    # Detokenize the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return {"generated_text": generated_text}
