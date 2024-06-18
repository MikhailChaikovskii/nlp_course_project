import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the fine-tuned model and tokenizer

from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
path = Path("./chat_bot/models/fine-tuned-dialogpt_new")
#path.absolute()
tokenizer_tuned = AutoTokenizer.from_pretrained(path.absolute())
model_tuned = AutoModelForCausalLM.from_pretrained(path.absolute()).to(device)

def concatenate_latest_lines(dialogue_lines, tokenizer, max_length=400):
    concatenated = ""
    for line in reversed(dialogue_lines):
        concatenated = line + tokenizer.eos_token + concatenated
        if len(tokenizer.encode(concatenated)) > max_length:
            break
    return concatenated

def generate_response(dialogue_lines, model, tokenizer, max_length=200, max_context_length=400):
    # Concatenate the latest lines of the dialogue
    context = concatenate_latest_lines(dialogue_lines,tokenizer,  max_context_length)

    # Tokenize and truncate context if necessary
    encoded_context = tokenizer.encode(context, return_tensors="pt").to(device)

    # Generate a response
    response_ids = model.generate(encoded_context, eos_token_id = tokenizer.eos_token_id, max_length=encoded_context.shape[1] + max_length, pad_token_id=tokenizer.pad_token_id)

    # Extract the generated response (excluding the input)
    generated_response = response_ids[:, encoded_context.shape[1]:]

    # Decode the response
    response_text = tokenizer.decode(generated_response[0], skip_special_tokens=True)

    # Add new response to dialog
    dialogue_lines.append(response_text)

    return response_text

# Function to chat with the bot
def chat():
    print("Chat with the bot (type 'quit' to stop)!\n")
    dialogue_lines = []
    while True:
        user_input = input("User: ")
        # check if the user wants to quit
        if user_input.lower() == 'quit':
            break
        dialogue_lines.append(user_input)
        response= generate_response(dialogue_lines, model_tuned, tokenizer_tuned)
        print(f"Bot: {response}")

# Start the chat
chat()
