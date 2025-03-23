from datasets import load_dataset, load_from_disk
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import os

# --------------- Load and Preprocess Dataset ---------------

# Initialize tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# # Get the current directory dynamically
# base_dir = os.path.dirname(os.path.abspath(__file__))
# # Define dataset folder relative to the project directory
# dataset_dir = os.path.join(base_dir, "data\datasets", "rajpurkar/squad_v2")

# dataset = load_dataset("squad_v2", cache_dir=dataset_dir)

# # Reduce dataset size by 50% (for faster training or limited resources)
# train_dataset = dataset['train'].train_test_split(test_size=0.5, seed=42)['train']
# val_dataset = dataset['validation'].train_test_split(test_size=0.5, seed=42)['train']

# # Preprocessing function to tokenize and prepare inputs/labels
# def preprocess_function(examples):
#     inputs = ["Generate a question and answer from the context: " + context for context in examples['context']]
#     targets = [
#         f"Question: {question} Answer: {answer['text'][0] if answer['text'] else 'No answer'}"
#         for question, answer in zip(examples['question'], examples['answers'])
#     ]
#     model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
#     labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

# # Apply preprocessing to both train and validation sets
# train_dataset = train_dataset.map(preprocess_function, batched=True)
# val_dataset = val_dataset.map(preprocess_function, batched=True)

# # Save the processed datasets to disk
# save_dir = "./data/preprocessedDataset/processed_squad_v2"
# os.makedirs(save_dir, exist_ok=True)
# train_dataset.save_to_disk(os.path.join(save_dir, "train"))
# val_dataset.save_to_disk(os.path.join(save_dir, "validation"))

# --------------- Load the Processed Dataset ---------------
# If already processed and saved, you can skip this part

train_dataset = load_from_disk("./data/preprocessedDataset/processed_squad_v2/train")
val_dataset = load_from_disk("./data/preprocessedDataset/processed_squad_v2/validation")

# --------------- Define Training Arguments ---------------

training_args = TrainingArguments(
    output_dir="./training/results",              # Output directory for saving the model
    eval_strategy="epoch",                         # Evaluate after each epoch (changed to 'eval_strategy')
    learning_rate=2e-5,                            # Learning rate
    per_device_train_batch_size=8,                 # Training batch size
    per_device_eval_batch_size=8,                  # Evaluation batch size
    num_train_epochs=5,                            # Number of epochs
    weight_decay=0.01,                             # Weight decay for regularization
    logging_dir='./training/logs',                 # Log directory
    logging_steps=500,                             # Log every 500 steps
    report_to="none",                              # Disable logging to wandb (can be adjusted for other systems)
    save_total_limit=1,                            # Keep only the most recent model checkpoint
    save_steps=1000,                               # Save model every 1000 steps
    save_strategy="steps",                         # Save model based on steps
    run_name="T5_Squad_Training_Run",              # Name the training run
)

# --------------- Initialize Model ---------------

# Load the pre-trained T5-small model

model = T5ForConditionalGeneration.from_pretrained('t5-small')

# --------------- Data Collator ---------------
# Use a data collator for better batch processing during training

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# --------------- Initialize Trainer ---------------

trainer = Trainer(
    model=model,                         # The model to train
    args=training_args,                  # Training arguments
    train_dataset=train_dataset,         # The training dataset
    eval_dataset=val_dataset,            # The validation dataset
    tokenizer=tokenizer,                 # Tokenizer for padding and truncation
    data_collator=data_collator          # Data collator to handle padding during training
)

# --------------- Start Training ---------------
trainer.train()

# --------------- Save the Trained Model ---------------

# Save the final model and tokenizer

model.save_pretrained("./training/model/t5_squad_model")
tokenizer.save_pretrained("./training/model/t5_squad_model")

# Optionally, you can evaluate the model after training
trainer.evaluate()

