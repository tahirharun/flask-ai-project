from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model & tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Conversation memory
chat_history = []

# Maximum number of past messages to keep
MAX_MEMORY = 6  # last 3 user + 3 AI messages

# Hand-coded capitals
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

    # Normalize user message
    lower_msg = user_message.lower()

    # --- HAND-CODED CAPITAL CHECK ---
    if "capital" in lower_msg:
        # Look for 'of <country>'
        if "of" in lower_msg:
            country = lower_msg.split("of")[-1].strip().rstrip("?").title()
            if country in CAPITALS:
                return CAPITALS[country]
            else:
                return f"Sorry, I don't know the capital of {country}"

    # --- DYNAMIC GPT-2 RESPONSE ---
    chat_history.append(f"User: {user_message}")

    # Keep only last MAX_MEMORY messages
    memory = chat_history[-MAX_MEMORY:]

    # System instruction for GPT-2
    system_prompt = (
        "The following is a conversation between a helpful AI assistant and a user. "
        "The AI gives concise, friendly, and relevant responses."
    )

    # Build GPT-2 prompt
    prompt = system_prompt + "\n" + "\n".join(memory) + "\nAI:"

    # Encode input
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate output
    outputs = model.generate(
        inputs,
        max_length=len(inputs[0]) + 60,  # short responses
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=30,
        top_p=0.85,
        temperature=0.7
    )

    # Decode and extract AI response
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ai_response = text[len(prompt):].strip().split("\n")[0]

    # Add AI response to memory
    chat_history.append(f"AI: {ai_response}")

    return ai_response

def reset_chat():
    """Clear the conversation memory"""
    global chat_history
    chat_history = []
