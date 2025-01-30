import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from transformers import AutoTokenizer
import random
import os
from mlx_lm import load

def load_jsonl_dataset(file_path):
    """Loads a JSONL dataset and returns a list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def loss_fn(model, input_ids, target_ids):
    """Compute loss using cross-entropy."""
    logits = model(input_ids)  # Forward pass through the model
    return nn.losses.cross_entropy(logits, target_ids)  # Compute the loss

def start_fine_tuning():
    train_dataset_path = "/Users/prupro/Desktop/Github/PotterBot/qna_data/train.jsonl"
    val_dataset_path = "/Users/prupro/Desktop/Github/PotterBot/qna_data/val.jsonl"

    model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.2-4bit")

    # Load datasets
    train_dataset = load_jsonl_dataset(train_dataset_path)
    val_dataset = load_jsonl_dataset(val_dataset_path)

    # Define hyperparameters
    num_iters = 100
    steps_per_eval = 10
    learning_rate = 1e-5
    batch_size = 1  # Adjust based on memory constraints

    # Optimizer
    optimizer = optim.AdamW(learning_rate)

    # Training loop
    for step in range(num_iters):
        # Sample a random training batch
        batch = random.sample(train_dataset, batch_size)

        # Set pad_token if not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Or add a new pad token if you prefer

        # Tokenize the batch
        inputs = tokenizer.batch_encode_plus(
            [batch[0]["prompt"]],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="np"
        )

        # Tokenize the targets
        targets = tokenizer.batch_encode_plus(
            [batch[0]["completion"]],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="np"
        )

        # Convert to MLX tensors (use mx.tensor to convert the data)
        input_ids = mx.tensor(inputs["input_ids"], dtype=mx.float32)  # Specify dtype for proper tensor creation
        target_ids = mx.tensor(targets["input_ids"], dtype=mx.float32)

        # Forward pass and loss calculation
        loss = loss_fn(model, input_ids, target_ids)

        # Manually calculate gradients using the loss
        loss.backward()  # Backpropagate to compute gradients

        # Optimizer step (use MLX's built-in gradient updates)
        optimizer.step()

        # Validation loss calculation
        if step % steps_per_eval == 0:
            val_batch = random.sample(val_dataset, batch_size)
            val_inputs = tokenizer(val_batch[0]["prompt"], return_tensors="np", padding="max_length", truncation=True, max_length=256)
            val_targets = tokenizer(val_batch[0]["completion"], return_tensors="np", padding="max_length", truncation=True, max_length=256)

            val_input_ids = mx.tensor(val_inputs["input_ids"], dtype=mx.float32)
            val_target_ids = mx.tensor(val_targets["input_ids"], dtype=mx.float32)

            val_logits = model(val_input_ids)
            val_loss = nn.losses.cross_entropy(val_logits, val_target_ids)

            print(f"Step {step}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

    # Save the trained model
    save_dir = "Users/prupro/Desktop/Github/PotterBot/trained_models"
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    model.save(os.path.join(save_dir, "Mixtral_Managed"))
    print("Fine-tuning complete!")

if __name__ == "__main__":
    start_fine_tuning()
