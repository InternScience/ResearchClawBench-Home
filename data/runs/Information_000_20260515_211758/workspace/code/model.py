"""
Unified Autoregressive Framework with Decoupled Visual Encoding
for Multimodal Understanding (VQA) and Visual Generation (Text-to-Image)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

class DecoupledVisualEncoder(nn.Module):
    """Decoupled visual encoder for understanding and generation paths."""
    def __init__(self, embed_dim=768, num_heads=12, num_layers=6):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Shared visual patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        
        # Understanding-specific encoder
        self.understanding_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        
        # Generation-specific encoder (decoupled)
        self.generation_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        
        # Projection layers
        self.understand_proj = nn.Linear(embed_dim, embed_dim)
        self.generate_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, images, mode='understand'):
        # images: [B, C, H, W]
        patches = self.patch_embed(images)  # [B, embed_dim, H/16, W/16]
        B, C, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        if mode == 'understand':
            encoded = self.understanding_encoder(patches)
            return self.understand_proj(encoded)
        else:  # generate
            encoded = self.generation_encoder(patches)
            return self.generate_proj(encoded)

class UnifiedAutoregressiveTransformer(nn.Module):
    """Unified autoregressive Transformer with decoupled visual encoding."""
    def __init__(self, vocab_size=32000, embed_dim=768, num_layers=12, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.visual_encoder = DecoupledVisualEncoder(embed_dim, num_heads, num_layers//2)
        
        # Text embedding
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Unified Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        
        # Output heads
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.image_head = nn.Linear(embed_dim, 3 * 16 * 16)  # For image patch reconstruction
        
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, embed_dim))  # Max sequence length
        
    def forward(self, input_ids=None, images=None, mode='understand', attention_mask=None):
        B = 1
        if input_ids is not None:
            B = input_ids.shape[0]
            text_emb = self.text_embed(input_ids)
        else:
            text_emb = torch.zeros(B, 0, self.embed_dim, device=self.pos_embed.device)
            
        if images is not None:
            visual_emb = self.visual_encoder(images, mode=mode)
            # Concatenate text and visual embeddings
            if text_emb.shape[1] > 0:
                combined = torch.cat([text_emb, visual_emb], dim=1)
            else:
                combined = visual_emb
        else:
            combined = text_emb
            
        # Add positional embeddings
        seq_len = combined.shape[1]
        combined = combined + self.pos_embed[:, :seq_len, :]
        
        # Unified transformer
        hidden = self.transformer(combined)
        
        if mode == 'understand':
            # Language modeling for VQA
            logits = self.lm_head(hidden)
            return logits
        else:
            # Image generation head
            image_logits = self.image_head(hidden)
            return image_logits

def test_vqa(model, tokenizer, image_path, question):
    """Test visual question answering capability."""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)
    
    # Simple prompt
    prompt = f"Question: {question} Answer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, images=img_tensor, mode='understand')
        # Greedy decode last token
        pred_token = outputs[0, -1].argmax().item()
        answer = tokenizer.decode([pred_token])
    return answer

def test_text_to_image(model, prompt, tokenizer):
    """Test text-to-image generation capability."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        image_patches = model(input_ids=inputs.input_ids, mode='generate')
    return image_patches

if __name__ == "__main__":
    print("Model initialized successfully!")
    model = UnifiedAutoregressiveTransformer()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
