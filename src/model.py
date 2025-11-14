# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torchvision import models, transforms

# =========================
# Abstract Base Class
# =========================
class RecursiveNet(nn.Module, ABC):
    """
    Abstract class for recursive reasoning networks.
    It enforces the presence of:
      - input_embed: maps input (e.g. image) to embedding space
      - backbone: maps embedding -> embedding (same dimension)
      - output_head: maps embedding -> task-specific output (e.g. logits)
    """

    def __init__(self, state_dim=512, recursive_mode=True):
        super().__init__()
        self.STATE_DIM = state_dim
        self.recursive_mode = recursive_mode

        # --- Abstract components ---
        self.input_embed = None
        self.backbone = None
        self.output_head = None

    # ----- ABSTRACT METHODS -----
    @abstractmethod
    def get_input_embedding(self, raw_input: torch.Tensor) -> torch.Tensor:
        """Must be implemented by subclasses."""
        pass

    # ----- GENERIC METHODS -----
    def latent_recursion(self, input_embed, output_embed, latent_embed, n=6):
        """
        Performs latent recursive reasoning using only self.backbone.
        If recursive_mode=False, performs one forward transformation (no recursion).
        """
        if self.recursive_mode:
            for _ in range(n):
                fused_state = input_embed + output_embed + latent_embed
                latent_embed = self.backbone(fused_state)

            fused_state = latent_embed + input_embed
            output_embed = self.backbone(fused_state)
        else:
            # Straightforward mode: no recursive loop
            output_embed = self.backbone(input_embed)
        return output_embed, latent_embed

    def deep_recursion(self, raw_input, output_embed, latent_embed, n=6, T=3):
        """
        Performs one step of deep recursive reasoning over T time steps.
        If recursive_mode=False, just performs a single forward pass.
        """
        input_embed = self.get_input_embedding(raw_input)

        if self.recursive_mode:
            with torch.no_grad():
                for _ in range(T - 1):
                    output_embed, latent_embed = self.latent_recursion(
                        input_embed, output_embed, latent_embed, n
                    )

        # Final recursion (with gradient)
        output_embed, latent_embed = self.latent_recursion(
            input_embed, output_embed, latent_embed, n
        )

        logits = self.output_head(output_embed)
        return output_embed.detach(), latent_embed.detach(), logits

    def forward(self, raw_input, Nsup=16, n_latent=6, init_strategy="random"):
        """
        Forward pass for inference only.
        In recursive mode: multiple recursive refinements.
        In non-recursive mode: one forward pass.
        """
        assert not self.training, "Forward is for inference only."

        input_embed = self.get_input_embedding(raw_input)
        B, dev = raw_input.shape[0], raw_input.device

        # Initialize latent and output embeddings
        if init_strategy == "zeros":
            output_embed = torch.zeros(B, self.STATE_DIM, device=dev)
            latent_embed = torch.zeros(B, self.STATE_DIM, device=dev)
        else:  # "random"               
            output_embed = torch.randn(B, self.STATE_DIM, device=dev)
            latent_embed = torch.randn(B, self.STATE_DIM, device=dev)

        logits = None
        if self.recursive_mode:
            for _ in range(Nsup):
                output_embed, latent_embed, logits = self.deep_recursion(
                    raw_input, output_embed, latent_embed, n=n_latent, T=3
                )
        else:
            # Simple direct pass
            output_embed, latent_embed, logits = self.deep_recursion(
                raw_input, output_embed, latent_embed, n=n_latent, T=1
            )
        return logits


# =========================
# Concrete Implementation
# =========================
class SimpleCNN(RecursiveNet):
    """
    A simple CNN implementation of RecursiveNet.
    Flexible to various input sizes and number of classes.
    """

    def __init__(self, in_channels=1, num_classes=10, input_size=(28, 28), recursive_mode=True):
        super().__init__(state_dim=512, recursive_mode=recursive_mode)

        # --- CNN Feature Extractor ---
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # --- Dynamically compute flattened size ---
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_size)
            self.flattened_size = self.cnn_extractor(dummy).numel()

        # --- Input Embedding (CNN → 512D) ---
        self.input_embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.STATE_DIM),
            nn.ReLU(),
        )

        # --- Backbone (512 → 512) ---
        self.backbone = nn.Sequential(
            nn.Linear(self.STATE_DIM, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.STATE_DIM),
            nn.ReLU(),
        )

        # --- Output Head (512 → num_classes) ---
        self.output_head = nn.Linear(self.STATE_DIM, num_classes)

    def get_input_embedding(self, raw_input):
        """Compute the (B, 512) input embedding from raw images."""
        x = self.cnn_extractor(raw_input)
        input_embed = self.input_embed(x)
        return input_embed

class ResNet(RecursiveNet):
    """
    RecursiveNet variant using ResNet-18 as the input embedding extractor.
    ...
    """

    def __init__(self, num_classes=10, pretrained=False, recursive_mode=True, state_dim=512, 
                 use_precomputed_features=False): # <--- ADD ARG
        super().__init__(state_dim=state_dim, recursive_mode=recursive_mode)
        self.STATE_DIM = state_dim
        self.recursive_mode = recursive_mode
        self.pretrained = pretrained
        self.use_precomputed_features = use_precomputed_features # <--- ADD LINE

        if self.use_precomputed_features and not self.pretrained:
            raise ValueError("Cannot use_precomputed_features if model is not pretrained.")
            
        # --- Load ResNet18 backbone ---
        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Remove classifier head
        self.feature_extractor = nn.Sequential(*(list(resnet.children())[:-1]))  # until avgpool

        # Freeze feature extractor if pretrained
        if pretrained:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.feature_extractor.eval()  # ensure it stays in eval mode

        # --- Input embedding (ResNet -> 512D) ---
        self.input_embed = nn.Sequential(
            nn.Flatten(),               # (B, 512, 1, 1) -> (B, 512)
            nn.Linear(512, self.STATE_DIM), # <-- This input dim (512) is correct
            nn.ReLU(),
        )

        # --- Backbone (512 → 512) ---
        self.backbone = nn.Sequential(
            nn.Linear(self.STATE_DIM, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.STATE_DIM),
            nn.ReLU(),
        )

        # --- Output head (512 → num_classes) ---
        self.output_head = nn.Linear(self.STATE_DIM, num_classes)

        # --- Preprocessing for pretrained mode ---
        self.resize_transform = transforms.Resize((224, 224)) if pretrained else nn.Identity()

    # -------------------------------------------------------------------------
    # Freezing logic to ensure pretrained stays frozen even in model.train()
    # -------------------------------------------------------------------------
    def train(self, mode=True):
        """
        Override .train() to keep feature extractor frozen if pretrained.
        """
        super().train(mode)
        if self.pretrained:
            self.feature_extractor.eval()
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
        return self

    # -------------------------------------------------------------------------
    # Embedding computation
    # -------------------------------------------------------------------------
    def get_input_embedding(self, raw_input: torch.Tensor) -> torch.Tensor:
            """
            If self.use_precomputed_features:
                Assumes raw_input IS the precomputed feature (B, 512, 1, 1)
            Else:
                Preprocess raw_input (image), extract ResNet features,
                and map to 512D embedding.
            """
            # --- ADD THIS BLOCK ---
            if self.use_precomputed_features:
                # If features are precomputed, raw_input is (B, 512, 1, 1)
                # We just pass it to the input_embed module.
                input_embed = self.input_embed(raw_input)
                return input_embed
            # --- END OF BLOCK ---

            # --- Original logic (now in the 'else' case) ---
            # Handle grayscale input (B,1,H,W) → (B,3,H,W)
            if raw_input.shape[1] == 1:
                raw_input = raw_input.repeat(1, 3, 1, 1)

            # Resize if using pretrained backbone
            raw_input = self.resize_transform(raw_input)

            # Extract features
            if self.pretrained:
                with torch.no_grad():
                    x = self.feature_extractor(raw_input)  # (B, 512, 1, 1)
            else:
                x = self.feature_extractor(raw_input)  # (B, 512, 1, 1)
            
            input_embed = self.input_embed(x)
            return input_embed

    # -------------------------------------------------------------------------
    # Recursive reasoning methods
    # -------------------------------------------------------------------------
    def latent_recursion(self, input_embed, output_embed, latent_embed, n=6):
        if self.recursive_mode:
            for _ in range(n):
                fused = input_embed + output_embed + latent_embed
                latent_embed = self.backbone(fused)
            fused = latent_embed + input_embed
            output_embed = self.backbone(fused)
        else:
            output_embed = self.backbone(input_embed)
        return output_embed, latent_embed

    def deep_recursion(self, raw_input, output_embed, latent_embed, n=6, T=3):
        input_embed = self.get_input_embedding(raw_input)

        if self.recursive_mode:
            with torch.no_grad():
                for _ in range(T - 1):
                    output_embed, latent_embed = self.latent_recursion(
                        input_embed, output_embed, latent_embed, n
                    )

        output_embed, latent_embed = self.latent_recursion(
            input_embed, output_embed, latent_embed, n
        )
        logits = self.output_head(output_embed)
        return output_embed.detach(), latent_embed.detach(), logits
