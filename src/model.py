# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for MNIST,
    structured for recursive reasoning experiments.
    """
    def __init__(self, in_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 1. Embedding Layer (Image to Feature Map)
        # A small CNN to map the input image to a flattened feature vector.
        # Params: (1*9+1)*16 + (16*9+1)*32 = 160 + 4640 = 4,800
        self.embedding_layer = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # Output: (B, 16, 14, 14)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Output: (B, 32, 7, 7)
        )
        
        # Calculate the flattened feature size
        self.flattened_size = 32 * 7 * 7 # 1568

        # 2. Backbone (Feature Processing MLP)
        # This layer processes the features from the embedding layer.
        # Params: (1568*1024 + 1024) + (1024*128 + 128) = 1,606,656 + 131,200 = 1,737,856
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
        )
        
        # 3. Output Head (Classifier)
        # This layer takes the processed features and makes a classification.
        # Params: (128*10 + 10) = 1,290
        self.output_head = nn.Linear(128, num_classes)

    def forward(self, raw_input, Nsup=16):
        """
        Forwad pass only for inference
        args:
        raw_input: Input image tensor.
        Nsup: Number of deep supervision steps.
        """
        # Initial embedding
        assert self.training == False, "Forward method is only for inference."
        input_embed = self.embedding_layer(raw_input)
        output_embed = torch.empty_like(input_embed).normal_(mean=0.0, std=1.0)
        latent_embed = torch.empty_like(input_embed).normal_(mean=0.0, std=1.0)

        for step in range(Nsup):
            output_embed, latent_embed, output = self.deep_recursion(raw_input, output_embed, latent_embed, n=6, T=3)
        return output

    def latent_recursion(self, input_embed, output_embed, latent_embed,n=6):
        """
        A method to perform latent recursive reasoning.
        args:
        raw_input: Original input image tensor.
        output_embed: Previous output embedding tensor.
        latent_embed: Previous latent embedding tensor.
        n: Number of latent recursive reasoning steps.
        """
        for i in range(n):
            latent_embed = self.backbone(input_embed + output_embed + latent_embed)
        output_embed = self.backbone(latent_embed + input_embed)
        return output_embed, latent_embed
    
    def deep_recursion(self, raw_input, output_embed, latent_embed,n=6, T=3):
        """
        A method to perform deep recursive reasoning over T time steps.
        args:
        raw_input: Original input image tensor.
        output_embed: Initial output embedding tensor.
        latent_embed: Initial latent embedding tensor.
        n: Number of latent recursive reasoning steps per time step.
        T: Number of time steps for deep recursion.
        returns:
        output_embed: Final output embedding tensor.
        latent_embed: Final latent embedding tensor.
        logits: Classification logits tensor.
        """
        input_embed = self.embedding_layer(raw_input)
        with torch.no_grad():
            for j in range(T-1):
                output_embed, latent_embed = self.latent_recursion(input_embed, output_embed, latent_embed, n)
        # Final recursion with gradients
        output_embed, latent_embed = self.latent_recursion(input_embed, output_embed, latent_embed, n)
        logits = self.output_head(output_embed)
        return output_embed.detach(), latent_embed.detach(), logits