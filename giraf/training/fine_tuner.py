"""
Model fine-tuning using LoRA (Low-Rank Adaptation).
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    default_data_collator,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from accelerate import Accelerator


def fine_tune_model(train_data, base_model_name, tokenizer, val_data=None, 
                    save_dir="./finetuned_model", max_steps=100):
    """
    Fine-tune a base model with LoRA.
    
    Args:
        train_data: Training DataFrame
        base_model_name: Base model identifier
        tokenizer: Model tokenizer
        val_data: Validation DataFrame
        save_dir: Directory to save model
        max_steps: Maximum training steps
        
    Returns:
        str: Path to saved model
    """
    accelerator = Accelerator()
    print("Preprocessing the training dataset for fine-tuning...")
    
    # Convert to HF datasets
    hf_train_dataset = Dataset.from_pandas(train_data)
    hf_val_dataset = Dataset.from_pandas(val_data) if val_data is not None else None
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    
    def preprocess_function(batch):
        """Preprocess batch for training."""
        prompts = batch["kpi_input"]
        targets = batch["kpi_description"]
        
        full_texts = [p + "\n" + t + tokenizer.eos_token for p, t in zip(prompts, targets)]
        
        tokenized = tokenizer(
            full_texts,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        
        labels = []
        for i, text in enumerate(full_texts):
            input_ids = tokenized["input_ids"][i]
            label = list(input_ids)
            
            tokenized_prompt = tokenizer(prompts[i] + "\n", truncation=True, max_length=512)
            prompt_len = len(tokenized_prompt["input_ids"])
            
            for j in range(len(label)):
                if j < prompt_len or input_ids[j] == tokenizer.pad_token_id:
                    label[j] = -100
            labels.append(label)
        
        tokenized["labels"] = labels
        return tokenized
    
    processed_train_dataset = hf_train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=hf_train_dataset.column_names,
    )
    
    processed_val_dataset = None
    if hf_val_dataset is not None:
        processed_val_dataset = hf_val_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=hf_val_dataset.column_names,
        )
    
    print("Setting up data loaders...")
    train_loader = DataLoader(
        processed_train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    
    val_loader = None
    if processed_val_dataset is not None:
        val_loader = DataLoader(
            processed_val_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=default_data_collator,
        )
    
    print("Initializing base model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    print("Adding LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        inference_mode=False,
    )
    
    model = get_peft_model(base_model, lora_config)
    model.train()
    
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters found.")
    print(f"Trainable parameters: {len(trainable)}")
    
    optimizer = AdamW(model.parameters(), lr=2e-4)
    
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    
    best_loss = float("inf")
    early_stopping_triggered = False
    step_count = 0
    
    print("Starting fine-tuning...")
    for epoch in range(3):
        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"Epoch {epoch+1}, Step {step_count}, Loss {loss.item():.4f}")
            
            if step_count >= max_steps:
                print(f"Reached max training steps: {max_steps}")
                break
        
        if step_count >= max_steps:
            break
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")
        
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_loss - 1e-4:
                best_loss = avg_val_loss
                print(f"Validation loss improved to {avg_val_loss:.4f}")
            else:
                print("Early stopping triggered.")
                early_stopping_triggered = True
        
        if early_stopping_triggered:
            break
    
    save_path = f"{save_dir}/{base_model_name.replace('/', '_')}"
    print(f"Saving fine-tuned model to {save_path}...")
    accelerator.wait_for_everyone()
    model.save_pretrained(save_path, save_function=accelerator.save)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}.")
    
    return save_path
