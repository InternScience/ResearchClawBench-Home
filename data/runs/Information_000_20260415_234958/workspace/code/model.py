import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image

class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = GPT2LMHeadModel.from_pretrained("gpt2")
        self.text_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.proj = nn.Linear(self.vision_encoder.config.hidden_size, self.text_encoder.config.hidden_size)
        
    def forward(self, image, text):
        pass

