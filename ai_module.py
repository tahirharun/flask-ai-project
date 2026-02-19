from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

chat_history = []

MAX_MEMORY = 6

CAPITALS = {
    "France": "Paris",
    "Germany": "Berlin",
    "Kenya": "Nairobi",
    "Japan": "Tokyo",
    "United States": "Washington, D.C.",
    "Canada": "Ottawa",
    "India": "New Delhi",
    "Italy": "Rome",
    "Brazil": "Brasília",
    "Australia": "Canberra"
}

def get_ai_response(user_message):
    global chat_history

    lower_msg = user_message.lower()

    if "capital" in lower_msg:

        if "of" in lower_msg:
            country = lower_msg.split("of")[-1].strip().rstrip("?").title()
            if country in CAPITALS:
                return CAPITALS[country]
            else:
                return f"Sorry, I don't know the capital of {country}"

    chat_history.append(f"User: {user_message}")

    memory = chat_history[-MAX_MEMORY:]

    system_prompt = (
        "The following is a conversation between a helpful AI assistant and a user. "
        "The AI gives concise, friendly, and relevant responses."
    )

    prompt = system_prompt + "\n" + "\n".join(memory) + "\nAI:"

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_length=len(inputs[0]) + 60,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=30,
        top_p=0.85,
        temperature=0.7
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ai_response = text[len(prompt):].strip().split("\n")[0]

    chat_history.append(f"AI: {ai_response}")

    return ai_response

def reset_chat():
    """Clear the conversation memory"""
    global chat_history
    chat_history = []