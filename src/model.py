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
        
        # --- CONSTANT STATE DIMENSION ---
        # The input embedding, latent embedding, and output embedding must all be 512D.
        self.STATE_DIM = 512
        
        # 1. CNN Feature Extractor 
        # The initial part of the Embedding CNN, which outputs a feature map (B, 32, 7, 7).
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # Output: (B, 16, 14, 14)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Output: (B, 32, 7, 7)
        )
        
        # Calculate the flattened feature size
        self.flattened_size = 32 * 7 * 7 # 1568

        # 2. Input Embedding Projection (Completes the Embedding CNN)
        # Maps the 1568D flattened feature vector to the 512D input state (input_embed).
        self.input_embed_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.STATE_DIM), # Output is 512D
            nn.ReLU(),
        )
        
        # 3. Recursive Backbone (State Processor)
        # This is the SINGLE component used for ALL recursive state transformations.
        # It takes a 512D fused state and outputs a new 512D state (512D -> 512D).
        self.backbone = nn.Sequential(
            nn.Linear(self.STATE_DIM, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.STATE_DIM), # Output is 512D
            nn.ReLU(),
        )
        
        # 4. Output Head (Classifier)
        # This layer takes the final 512D output embedding and makes a classification.
        self.output_head = nn.Linear(self.STATE_DIM, num_classes)

    def get_input_embedding(self, raw_input):
        """Helper to compute the (B, 512) initial input embedding."""
        x = self.cnn_extractor(raw_input)
        input_embed = self.input_embed_projection(x)
        return input_embed

    def latent_recursion(self, input_embed, output_embed, latent_embed, n=6):
        """
        Performs latent recursive reasoning using only self.backbone.
        All inputs and outputs are (B, 512).
        """
        if n > 0:
            for i in range(n):
                # Fusing the 512D states via addition (input + output + latent)
                fused_state = input_embed + output_embed + latent_embed
                latent_embed = self.backbone(fused_state) # 512D -> 512D
                
            # Fusing the 512D states (latent + input) to compute the new output
            fused_state = latent_embed + input_embed
            output_embed = self.backbone(fused_state) # 512D -> 512D
        else :
            # If n=0, skip recursion and just return inputs
            fused_state = input_embed + output_embed + latent_embed
            output_embed = self.backbone(fused_state)
        return output_embed, latent_embed
    
    def deep_recursion(self, raw_input, output_embed, latent_embed,n=6, T=3):
        """
        Performs one step of deep recursive reasoning over T time steps.
        """
        # Get the 512D input embedding
        input_embed = self.get_input_embedding(raw_input)
        
        with torch.no_grad():
            for j in range(T-1):
                output_embed, latent_embed = self.latent_recursion(input_embed, output_embed, latent_embed, n)
        
        # Final recursion with gradients
        output_embed, latent_embed = self.latent_recursion(input_embed, output_embed, latent_embed, n)
        
        # Classify the final 512D output embedding
        logits = self.output_head(output_embed)
        
        return output_embed.detach(), latent_embed.detach(), logits

    def forward(self, raw_input, Nsup=16, n_latent=6):
        """
        Forward pass for inference.
        """
        assert not self.training, "Forward method is only for inference."
        
        input_embed = self.get_input_embedding(raw_input)
        B = raw_input.shape[0]
        dev = raw_input.device

        # Initialize state embeddings to the correct (B, 512) shape
        output_embed = torch.randn(B, self.STATE_DIM, device=dev)
        latent_embed = torch.randn(B, self.STATE_DIM, device=dev)

        logits = None
        for _ in range(Nsup):
            output_embed, latent_embed, logits = self.deep_recursion(
                raw_input, output_embed, latent_embed, n=n_latent, T=3
            )
        return logits