import pandas as pd
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

#Load CSV data and convert to text
def load_and_convert_csv_to_text(csv_file):
    df = pd.read_csv(csv_file)
    text_data = []

    for index, row in df.iterrows():
        text = (
            f"In {row['Market']}, the cost of marketing is ${row['Marketing']}, "
            f"utility is ${row['Utility']}, authentication is ${row['Authentication']}, "
            f"and service is ${row['Service']}."
        )
        # Authentication-International
        if pd.notna(row['Authentication-International']):
            text += f" The international authentication rate is ${row['Authentication-International']}."
        
        text_data.append(text)
    
    return text_data

csv_file = '461319948_481132811590294_6041648812712871532_n.csv'  # Replace with your CSV file path
text_data = load_and_convert_csv_to_text(csv_file)

#Tokenize the data
tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-3.2')

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True)

# Convert text data into a Hugging Face dataset
dataset = Dataset.from_dict({"text": text_data})
tokenized_dataset = dataset.map(preprocess_function, batched=True)


model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-3.2')

# Fine-tune the model
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=2,  
    num_train_epochs=3,  
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',  
    logging_steps=200,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()


model.save_pretrained('./fine_tuned_llama')

# Step 8: Generate text from the fine-tuned model
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage after fine-tuning
prompt = "What is the cost of authentication in India as of July 2024?"
response = generate_response(prompt)
print(response)
