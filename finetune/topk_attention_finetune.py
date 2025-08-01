#!/usr/bin/env python3
"""
TopK Attention Fine-tuning for Qwen2.5-0.5B

This script implements fine-tuning of attention weights to approximate topk attention.
Supports two training objectives:
1. Minimize perplexity on training data
2. Recover original attention scores after softmax

Key Features:
- Per-head TopK sparsification for Q and K projections
- Proper causal masking maintained
- Support for baseline (dense) training when topk=None
"""

import os
# Set environment variable to suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm
import logging
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import wandb
import math

# Import model-specific functions for rotary embeddings
try:
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv, eager_attention_forward
except ImportError:
    print("Warning: Could not import Qwen2 functions")
    apply_rotary_pos_emb = None
    repeat_kv = None
    eager_attention_forward = None

try:
    from transformers.cache_utils import Cache
except ImportError:
    print("Warning: Could not import Cache from transformers")
    Cache = None


@dataclass
class TopKAttentionConfig:
    """Configuration for TopK attention fine-tuning"""
    model_name: str = "Qwen/Qwen2.5-0.5B"
    topk: int = None  # None for baseline (dense Q/K training), int for TopK sparsification
    objective: str = "ppl"  # "ppl" or "attn"
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_length: int = 1024
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    output_dir: str = "./topk_attention_output"
    logging_steps: int = 5
    save_steps: int = 5000
    eval_steps: int = 500
    warmup_steps: int = 200
    weight_decay: float = 0.01
    use_wandb: bool = False
    wandb_project: str = "topk-attention"


class TopKFunction(torch.autograd.Function):
    """Custom TopK function with straight-through estimator for gradients"""
    
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, k: int) -> torch.Tensor:
        """Forward pass: keep top k values, zero out others"""
        # If k >= dimension, return input as-is (no sparsification needed)
        last_dim = input_tensor.size(-1)
        if k >= last_dim:
            # Save empty indices for backward pass
            ctx.save_for_backward(torch.empty(0, device=input_tensor.device, dtype=torch.long))
            ctx.input_shape = input_tensor.shape
            ctx.full_tensor = True
            return input_tensor
        
        # Get topk values and indices
        topk_values, topk_indices = torch.topk(input_tensor, k, dim=-1)
        
        # Create output tensor with zeros
        output = torch.zeros_like(input_tensor)
        
        # Scatter topk values back to their positions
        output.scatter_(-1, topk_indices, topk_values)
        
        # Save indices for backward pass
        ctx.save_for_backward(topk_indices)
        ctx.input_shape = input_tensor.shape
        ctx.full_tensor = False
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass: straight-through estimator"""
        topk_indices, = ctx.saved_tensors
        
        # If full tensor was returned (k >= dimension), pass all gradients
        if ctx.full_tensor:
            return grad_output, None
        
        # Create gradient tensor
        grad_input = torch.zeros(ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype)
        
        # Only propagate gradients for topk positions
        grad_input.scatter_(-1, topk_indices, grad_output.gather(-1, topk_indices))
        
        return grad_input, None


def topk_sparse(x: torch.Tensor, k: int) -> torch.Tensor:
    """Apply topk sparsification"""
    return TopKFunction.apply(x, k)


class TopKProjection(nn.Module):
    """Wrapper for linear projection with per-head TopK sparsification"""
    def __init__(self, original_proj, topk, num_heads, head_dim):
        super().__init__()
        self.original_proj = original_proj
        self.topk = topk
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Calculate effective TopK per head
        self.effective_topk = min(topk, head_dim) if topk is not None else None
        
        # Make original_proj parameters trainable
        for param in self.original_proj.parameters():
            param.requires_grad = True
            
        print(f"TopKProjection: topk={topk}, num_heads={num_heads}, head_dim={head_dim}, effective_topk={self.effective_topk}")
    
    def forward(self, x):
        output = self.original_proj(x)
        
        # If no TopK or effective TopK >= head_dim, return as-is
        if self.effective_topk is None or self.effective_topk >= self.head_dim:
            return output
        
        # Apply TopK per head
        batch_size, seq_len, total_dim = output.shape
        
        # Reshape to separate heads: [batch, seq, num_heads, head_dim]
        output_heads = output.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply TopK to each head independently
        output_sparse = topk_sparse(output_heads, self.effective_topk)
        
        # Reshape back to original dimensions
        output_sparse = output_sparse.view(batch_size, seq_len, total_dim)
        
        return output_sparse
    
    @property
    def weight(self):
        return self.original_proj.weight
    
    @property
    def bias(self):
        return self.original_proj.bias
    
    @property
    def in_features(self):
        return self.original_proj.in_features
    
    @property
    def out_features(self):
        return self.original_proj.out_features


class QwenTopKAttention(nn.Module):
    """TopK Attention for Qwen models - with per-head sparsification"""
    def __init__(self, original_attention, topk):
        super().__init__()
        # Copy all attributes from original attention
        self.config = original_attention.config
        self.layer_idx = getattr(original_attention, 'layer_idx', None)
        self.attention_dropout = original_attention.attention_dropout
        self.hidden_size = original_attention.config.hidden_size
        self.num_heads = original_attention.config.num_attention_heads
        self.head_dim = original_attention.head_dim
        self.num_key_value_heads = original_attention.config.num_key_value_heads
        self.num_key_value_groups = original_attention.num_key_value_groups
        self.max_position_embeddings = original_attention.config.max_position_embeddings
        self.rope_theta = original_attention.config.rope_theta
        self.is_causal = True
        self.scaling = original_attention.scaling
        
        # Create TopK projections for Q and K with proper head dimensions
        if topk is not None:
            self.q_proj = TopKProjection(
                original_attention.q_proj, 
                topk, 
                self.num_heads, 
                self.head_dim
            )
            self.k_proj = TopKProjection(
                original_attention.k_proj, 
                topk, 
                self.num_key_value_heads, 
                self.head_dim
            )
        else:
            # Baseline mode: make Q and K trainable but no TopK
            self.q_proj = original_attention.q_proj
            self.k_proj = original_attention.k_proj
            for param in self.q_proj.parameters():
                param.requires_grad = True
            for param in self.k_proj.parameters():
                param.requires_grad = True
        
        # Keep original V and O projections (frozen)
        self.v_proj = original_attention.v_proj
        self.o_proj = original_attention.o_proj
        
        # Freeze v_proj and o_proj
        for param in self.v_proj.parameters():
            param.requires_grad = False
        for param in self.o_proj.parameters():
            param.requires_grad = False
        
        print(f"Created QwenTopKAttention with TopK={topk}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass that matches new Qwen2Attention interface"""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # print(f"--- Layer {self.layer_idx} Debugging ---")
        # print(f"hidden_states.requires_grad: {hidden_states.requires_grad}")
        # print(f"hidden_states.grad_fn: {hidden_states.grad_fn}")

        # Apply (potentially TopK) Q and K projections
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # print(f"query_states (after q_proj) requires_grad: {query_states.requires_grad}")

        # Apply rotary position embedding
        cos, sin = position_embeddings
        if apply_rotary_pos_emb is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            # print(f"query_states (after RoPE) requires_grad: {query_states.requires_grad}")
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Use eager attention forward with proper causal masking
        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def create_topk_model(model,model_name:str, topk: Optional[int] = None):
    """Create a model with TopK attention layers"""
    
    # Load model
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float32,  # Use float32 for stability
    #     device_map="auto"
    # )
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Determine head_dim for the model
    first_attention = None
    for name, module in model.named_modules():
        if hasattr(module, 'self_attn'):
            first_attention = module.self_attn
            break
    
    if first_attention is None:
        raise ValueError("Could not find attention layers in model")
    
    head_dim = first_attention.head_dim
    print(f"Model head_dim: {head_dim}")
    
    # Check if this should be treated as baseline mode
    if topk is None or topk >= head_dim:
        if topk is not None and topk >= head_dim:
            print(f"TopK={topk} >= head_dim={head_dim}, treating as baseline mode")
        else:
            print("Baseline mode: making only Q and K projections trainable")
        
        trainable_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'self_attn'):
                # Make Q and K projections trainable
                for param in module.self_attn.q_proj.parameters():
                    param.requires_grad = True
                    trainable_count += param.numel()
                for param in module.self_attn.k_proj.parameters():
                    param.requires_grad = True
                    trainable_count += param.numel()
        print(f"Made Q and K projections trainable ({trainable_count:,} parameters)")
    else:
        # TopK mode: replace attention layers
        replaced_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'self_attn'):
                if 'qwen' in model_name.lower():
                    # Replace with QwenTopKAttention
                    module.self_attn = QwenTopKAttention(module.self_attn, topk)
                    replaced_count += 1
        
        print(f"Replaced {replaced_count} attention layers with TopK attention")
    
    # Count and log trainable parameters
    trainable_params = []
    total_trainable = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.shape, param.numel()))
            total_trainable += param.numel()
    
    print(f"\nTrainable parameters ({total_trainable:,} total):")
    for name, shape, numel in trainable_params:
        print(f"  {name}: {shape} ({numel:,} params)")
    
    return model


def get_dataset(config: TopKAttentionConfig, tokenizer):
    """Load and prepare dataset"""
    # Load dataset
    dataset = load_dataset(config.dataset_name, config.dataset_config)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_length,
            padding="max_length",
            return_tensors="pt"
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset


class TopKAttentionTrainer(Trainer):
    """Custom trainer that supports both perplexity and attention recovery objectives"""
    
    def __init__(self, objective="ppl", original_model=None, **kwargs):
        super().__init__(**kwargs)
        self.objective = objective
        self.original_model = original_model
        
        if self.objective == "attn" and self.original_model is None:
            raise ValueError("original_model is required for attention recovery objective")
        
        # Move original model to same device as training model
        if self.original_model is not None:
            self.original_model.to(self.model.device)
            self.original_model.eval()
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss computation based on objective"""
        
        if self.objective == "ppl":
            # Standard language modeling loss
            outputs = model(**inputs)
            loss = outputs.loss
            # # Scale loss by gradient accumulation steps for proper averaging
            # if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            
            return (loss, outputs) if return_outputs else loss
        
        elif self.objective == "attn":
            # Attention recovery loss
            
            # Get attention weights from original model
            with torch.no_grad():
                original_outputs = self.original_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    output_attentions=True
                )
                original_attentions = original_outputs.attentions  # List of [batch, heads, seq, seq]
            
            # Get attention weights from TopK model
            topk_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_attentions=True
            )
            topk_attentions = topk_outputs.attentions  # List of [batch, heads, seq, seq]
            
            # Compute MSE loss between attention weights
            attention_loss = 0.0
            num_layers = len(original_attentions)
            
            for i in range(num_layers):
                # MSE loss between attention matrices
                layer_loss = F.mse_loss(topk_attentions[i], original_attentions[i])
                attention_loss += layer_loss
            
            # Average over layers
            loss = attention_loss / num_layers
            
            return (loss, topk_outputs) if return_outputs else loss
        
        else:
            raise ValueError(f"Unknown objective: {self.objective}")


def main():
    parser = argparse.ArgumentParser(description="TopK Attention Fine-tuning")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--topk", type=int, default=None, help="TopK sparsification (None for baseline)")
    parser.add_argument("--objective", type=str, default="ppl", choices=["ppl", "attn"])
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    # Set output directory based on configuration
    if args.output_dir is None:
        if args.topk is not None:
            args.output_dir = f"./topk{args.topk}_{args.objective}_training"
        else:
            args.output_dir = f"./baseline_{args.objective}_training"
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create config
    config = TopKAttentionConfig(
        model_name=args.model_name,
        topk=args.topk,
        objective=args.objective,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        output_dir=args.output_dir,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        use_wandb=args.use_wandb
    )
    
    logger.info(f"Configuration: {config}")
    
    # Initialize wandb if requested
    if config.use_wandb:
        # Extract model name (e.g., "Qwen2.5-0.5B" from "Qwen/Qwen2.5-0.5B")
        model_short_name = config.model_name.split('/')[-1]
        wandb.init(
            project=config.wandb_project,
            config=vars(config),
            name=f"{model_short_name}_topk{config.topk}_{config.objective}" if config.topk else f"{model_short_name}_baseline_{config.objective}"
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    model = create_topk_model(config.model_name, config.topk)
    
    # Load dataset
    dataset = get_dataset(config, tokenizer)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["wandb"] if config.use_wandb else [],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        save_total_limit=3,
        fp16=False,  # Use FP32 for stability
        gradient_checkpointing=False,  # Disable for stability
        logging_first_step=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Load original model if needed for attention recovery
    original_model = None
    if config.objective == "attn":
        logger.info("Loading original model for attention recovery...")
        original_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        # Freeze all parameters
        for param in original_model.parameters():
            param.requires_grad = False
        logger.info("Original model loaded and frozen.")
    
    # Create trainer
    trainer = TopKAttentionTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        objective=config.objective,
        original_model=original_model,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    
    # Final evaluation
    logger.info("Final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
