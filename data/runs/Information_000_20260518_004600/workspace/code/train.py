"""
Training Script for Unified Framework
======================================
Training pipeline for both the unified framework and baseline models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple
import sys
sys.path.append(str(Path(__file__).parent))

from unified_framework import (
    FrameworkConfig, UnifiedFramework, SingleEncoderBaseline, create_model
)


class MultimodalDataset(Dataset):
    """Dataset for multimodal training."""
    
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, 0  # Dummy label


class UnderstandingLoss(nn.Module):
    """Loss for understanding tasks (VQA, captioning)."""
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))


class GenerationLoss(nn.Module):
    """Loss for generation tasks (next token prediction)."""
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))


class ReconstructionLoss(nn.Module):
    """Loss for image reconstruction (VQ loss)."""
    
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return self.l1_loss(reconstructed, original)


class Trainer:
    """Training manager for unified framework."""
    
    def __init__(
        self,
        model: nn.Module,
        config: FrameworkConfig,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # Loss functions
        self.understanding_loss = UnderstandingLoss()
        self.generation_loss = GenerationLoss()
        self.reconstruction_loss = ReconstructionLoss()
    
    def train_step(
        self,
        images: torch.Tensor,
        task_type: str = "understanding",
        text_tokens: torch.Tensor = None
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        images = images.to(self.device)
        if text_tokens is not None:
            text_tokens = text_tokens.to(self.device)
        
        # Forward pass
        if isinstance(self.model, UnifiedFramework):
            outputs = self.model(
                images,
                text_tokens=text_tokens,
                task_type=task_type
            )
        else:
            outputs = self.model(images, text_tokens=text_tokens, task_type=task_type)
        
        # Compute losses
        total_loss = 0
        loss_dict = {}
        
        # VQ loss
        if "vq_loss" in outputs:
            vq_loss = outputs["vq_loss"]
            total_loss += vq_loss
            loss_dict["vq_loss"] = vq_loss.item()
        
        # Reconstruction loss (for generation task)
        if task_type == "generation" and "generation_logits" in outputs:
            # Create dummy target for generation
            gen_targets = torch.randint(0, self.config.codebook_size, 
                                       (images.shape[0], outputs["generation_logits"].shape[1]),
                                       device=self.device)
            gen_loss = self.generation_loss(outputs["generation_logits"], gen_targets)
            total_loss += gen_loss
            loss_dict["generation_loss"] = gen_loss.item()
        
        # Understanding loss
        if text_tokens is not None and "understanding_logits" in outputs:
            # Create dummy target for understanding
            understanding_targets = torch.randint(0, self.config.vocab_size,
                                                 (images.shape[0], outputs["understanding_logits"].shape[1] - 1),
                                                 device=self.device)
            understanding_loss = self.understanding_loss(
                outputs["understanding_logits"][:, :-1, :],
                understanding_targets
            )
            total_loss += understanding_loss
            loss_dict["understanding_loss"] = understanding_loss.item()
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        loss_dict["total_loss"] = total_loss.item()
        return loss_dict
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 10,
        task_schedule: List[str] = None
    ) -> List[Dict[str, float]]:
        """Full training loop."""
        if task_schedule is None:
            task_schedule = ["understanding", "generation", "mixed"]
        
        history = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, (images, _) in enumerate(dataloader):
                # Cycle through task types
                task_type = task_schedule[batch_idx % len(task_schedule)]
                
                # Training step
                loss_dict = self.train_step(images, task_type=task_type)
                epoch_loss += loss_dict["total_loss"]
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                          f"Task: {task_type}, Loss: {loss_dict['total_loss']:.4f}")
            
            # Update scheduler
            self.scheduler.step()
            
            # Record epoch statistics
            avg_loss = epoch_loss / max(num_batches, 1)
            history.append({
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "learning_rate": self.scheduler.get_last_lr()[0]
            })
            
            print(f"\nEpoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}\n")
        
        return history


def create_synthetic_data(num_samples: int = 100) -> List[str]:
    """Create synthetic data paths for demonstration."""
    # In real scenario, these would be actual image paths
    data_dir = Path("data")
    image_paths = list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg"))
    
    # Repeat to create enough samples
    if len(image_paths) < num_samples:
        image_paths = image_paths * (num_samples // len(image_paths) + 1)
    
    return image_paths[:num_samples]


def train_models():
    """Train both unified and baseline models."""
    # Configuration
    config = FrameworkConfig(
        hidden_dim=256,  # Reduced for faster training
        num_heads=4,
        num_layers=3,
        feedforward_dim=512,
        max_seq_len=128
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data
    image_paths = create_synthetic_data(50)
    dataset = MultimodalDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    results = {}
    
    # Train Unified Framework
    print("\n" + "="*60)
    print("Training Unified Framework (Decoupled Visual Encoding)")
    print("="*60)
    
    unified_model = create_model("unified", config)
    unified_trainer = Trainer(unified_model, config, device)
    unified_history = unified_trainer.train(dataloader, num_epochs=5)
    results["unified"] = unified_history
    
    # Train Baseline (Single Encoder)
    print("\n" + "="*60)
    print("Training Baseline (Single Visual Encoder)")
    print("="*60)
    
    baseline_model = create_model("baseline", config)
    baseline_trainer = Trainer(baseline_model, config, device)
    baseline_history = baseline_trainer.train(dataloader, num_epochs=5)
    results["baseline"] = baseline_history
    
    # Save results
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    with open(outputs_dir / "training_history.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining completed! Results saved to outputs/training_history.json")
    
    return results, unified_model, baseline_model


if __name__ == "__main__":
    results, unified_model, baseline_model = train_models()