import ollama
import torch
import logging
import json
import os
import re
import datetime 

#================================================================================================
# Memory configuration
#================================================================================================

class BotMemory:
    def __init__(self, max_history=10, memory_file="C:\\Ai\\guffy\\memory\\bot_memory.json"):
        self.messages = []  # Stores the conversation history
        self.max_history = max_history  # Maximum number of messages to keep in memory
        self.memory_file = memory_file  # File to save/load memory
        
        # Load existing memory from file, if it exists
        self.load_memory()

    def add_message(self, role, content):
        """Add a new message to the conversation history with a timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p")
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })
        # Keep only the last `max_history` messages
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
        # Save memory to file after each update
        self.save_memory()

    def get_memory(self):
        """Retrieve the current conversation history."""
        return self.messages
    
    def save_memory(self):
        """Save the current conversation history to a file."""
        try:
            with open(self.memory_file, 'w') as file:
                json.dump(self.messages, file, indent=4)
        except IOError as e:
            print(f"Error writing to file: {e}")
        except TypeError as e:
            print(f"Error serializing messages to JSON: {e}")
    
    def load_memory(self):
        """Load the conversation history from a file, if it exists."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as file:
                    self.messages = json.load(file)
                    # Ensure all messages have a timestamp (for backward compatibility)
                    for message in self.messages:
                        if "timestamp" not in message:
                            message["timestamp"] = "Unknown"
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error loading memory from file: {e}")
                self.messages = []

#================================================================================================
# Configuring the AI Assistant
#================================================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set log level
    format="%(message)s",  # Format logs
    filename="C:\\Ai\\guffy\\logs\\chatbot.log",  # Save logs to a file
    filemode="a"  # Append to the log file
)

if torch.cuda.is_available():
    device = f"Using CUDA GPU: {torch.cuda.get_device_name(0)}"
else:
    device = "Using CPU"
    
print(device)

# Personality prompt for the assistant
def get_personality_prompt():
    import datetime
    # Get current time and day
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    current_day = datetime.datetime.now().strftime("%A, %B %d, %Y")
    return (
        f"Always respond in dialogue only, without any actions, stage directions, or physical descriptions. "
        f"DO NOT IN ANY CIRCUMSTANCES DESCRIBE YOUR PHYSICAL MOVEMENTS OR ACTIONS. "
        f"Your name is Luna. "
        f"The current time is {current_time} on {current_day}. Use this information when relevant to provide context-aware responses, especially for time- or day-related queries."
    )


print("Your AI Assistant (Gemma 2 9b) is now running. (Type a goodbye message like 'bye', 'goodbye', or 'see ya' to quit.)")

#================================================================================================
# Conversation history and main chat loop
#================================================================================================

# Get current personality prompt with time and day
personality_prompt = get_personality_prompt()

# Create message history (outside loop to maintain context)
messages = [
    {"role": "system", "content": personality_prompt}  # Set AI behavior
]

# Initialize BotMemory with a maximum history and memory file
bot_memory = BotMemory(max_history=50, memory_file="C:\\Ai\\guffy\\memory\\bot_memory.json")

# List of goodbye phrases to detect in user input and AI response (case-insensitive)
goodbye_phrases = ["bye", "goodbye", "see ya", "see you", "exit", "quit", "later", "adios", "farewell"]

# Main chat loop
while True:

    #================================================================================================
    # User input 
    #================================================================================================

    user_input = input("\nYou: ")
    
    # Log user input
    logging.info(f"User: {user_input}")

    # Add user input to memory
    bot_memory.add_message("user", user_input)

    # Update messages with the current memory
    messages = [{"role": "system", "content": personality_prompt}] + bot_memory.get_memory()

    #================================================================================================
    # AI response
    #================================================================================================

    try:
        # Generate response using Ollama's Gemma 2 (9B) model
        response = ollama.chat(model="gemma2:9b", messages=messages)
        
        # Extract and clean the text response
        assistant_reply = response['message']['content']
        assistant_reply = re.sub(r'\\n', ' ', assistant_reply)  # Replace literal '\n' with space
        assistant_reply = re.sub(r'\n\s*\n', '\n', assistant_reply)  # Replace multiple newlines with single newline
        assistant_reply = assistant_reply.strip()  # Remove leading/trailing whitespace

        # Check if user input contains a goodbye phrase
        is_goodbye = any(phrase in user_input.lower() for phrase in goodbye_phrases)
        
        if is_goodbye:
            # Check if AI response indicates a farewell
            if any(phrase in assistant_reply.lower() for phrase in goodbye_phrases):
                # AI is confirming goodbye, so end the conversation
                logging.info(f"Luna: {assistant_reply}")  # Log farewell response
                bot_memory.add_message("assistant", assistant_reply)  # Save farewell to memory
                print("Luna:", assistant_reply)  # Display farewell
                print("Goodbye!")  # Final message before exiting
                break
            else:
                # AI didn't confirm goodbye, so continue the conversation
                logging.info(f"Luna: {assistant_reply}")  # Log response
                bot_memory.add_message("assistant", assistant_reply)  # Save response to memory
                print("Luna:", assistant_reply)
                continue

        # Normal response (no goodbye detected in user input)
        logging.info(f"Luna: {assistant_reply}")  # Log response
        bot_memory.add_message("assistant", assistant_reply)  # Save response to memory
        print("Luna:", assistant_reply)

    except Exception as e:
        logging.error(f"Error generating response: {e}")
        print("Luna: Oops! Something went wrong.")