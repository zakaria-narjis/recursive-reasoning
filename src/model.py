# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torchvision import models, transforms
import math

# Import components from our modular vit.py
from src.vit import TransformerEncoder

# =========================
# Abstract Base Class
# =========================
class RecursiveNet(nn.Module, ABC):
    """
    Abstract class for recursive reasoning networks.
    """

    def __init__(self, state_dim=512, recursive_mode=True):
        super().__init__()
        # state_dim can be int (for CNN/MLP) or tuple (S, D) for ViT
        self.STATE_DIM = state_dim
        self.recursive_mode = recursive_mode

        # --- Abstract components ---
        self.input_embed = None
        self.backbone = None
        self.output_head = None

    @abstractmethod
    def get_input_embedding(self, raw_input: torch.Tensor) -> torch.Tensor:
        pass

    def latent_recursion(self, input_embed, output_embed, latent_embed, n=6):
        if self.recursive_mode:
            for _ in range(n):
                # Element-wise addition works for both (B, D) and (B, S, D)
                fused_state = input_embed + output_embed + latent_embed
                latent_embed = self.backbone(fused_state)

            fused_state = latent_embed + input_embed
            output_embed = self.backbone(fused_state)
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

    def forward(self, raw_input, Nsup=16, n_latent=6, init_strategy="random"):
        assert not self.training, "Forward is for inference only."
        B, dev = raw_input.shape[0], raw_input.device

        # === REPLACE THIS BLOCK (Initialization logic) ===
        # Handle both int (CNN) and tuple (ViT) state dimensions
        if isinstance(self.STATE_DIM, int):
            shape = (B, self.STATE_DIM)
        else:
            shape = (B, *self.STATE_DIM) # Unpack tuple (Seq_Len, Dim)

        if init_strategy == "zeros":
            output_embed = torch.zeros(*shape, device=dev)
            latent_embed = torch.zeros(*shape, device=dev)
        else:  # "random"               
            output_embed = torch.randn(*shape, device=dev)
            latent_embed = torch.randn(*shape, device=dev)
        # ==================================================

        logits = None
        if self.recursive_mode:
            for _ in range(Nsup):
                output_embed, latent_embed, logits = self.deep_recursion(
                    raw_input, output_embed, latent_embed, n=n_latent, T=3
                )
        else:
            output_embed, latent_embed, logits = self.deep_recursion(
                raw_input, output_embed, latent_embed, n=n_latent, T=1
            )
        return logits


class ViTClassificationHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.head_layer = nn.Linear(dim, num_classes)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Extracts CLS token from state and classifies.
        Input: (B, Seq_Len, Dim)
        Output: (B, Num_Classes)
        """
        # Extract CLS token (index 0)
        cls_embed = state[:, 0]
        return self.head_layer(cls_embed)
    
class RecursiveViT(RecursiveNet):
    """
    RecursiveNet using a Vision Transformer (ViT) architecture.
    State is kept as 3D tensor: (B, Seq_Len, Dim).
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=512,            # hidden_dim
        depth=6,            
        heads=8,
        mlp_dim=2048,
        recursive_mode=True,
        channels=3
    ):
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        seq_length = num_patches + 1 # +1 for Class Token
        
        # Set STATE_DIM to a tuple so Trainer knows it's 3D
        state_dim = (seq_length, dim)
        
        super().__init__(state_dim=state_dim, recursive_mode=recursive_mode)
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.seq_length = seq_length

        # --- Input Embedding: (B, C, H, W) -> (B, S, D) ---
        self.patch_to_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, dim))
        
        # --- Backbone: Transformer Encoder (B, S, D) -> (B, S, D) ---
        self.backbone = TransformerEncoder(
            num_layers=depth,
            num_heads=heads,
            hidden_dim=dim,
            mlp_dim=mlp_dim,
            dropout=0.1,
            attention_dropout=0.1
        )
        
        self.output_head = ViTClassificationHead(dim, num_classes)
        
        self._init_vit_weights()

    def _init_vit_weights(self):
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def get_input_embedding(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        Converts (B, C, H, W) -> (B, Seq_Len, Dim)
        """
        b, c, h, w = raw_input.shape
        
        # 1. Patchify: (B, C, H, W) -> (B, Dim, GridH, GridW)
        x = self.patch_to_embedding(raw_input)
        
        # 2. Flatten patches: (B, Dim, GridH, GridW) -> (B, Dim, N_Patches)
        x = x.flatten(2)
        
        # 3. Transpose: (B, Dim, N_Patches) -> (B, N_Patches, Dim)
        x = x.transpose(1, 2)
        
        # 4. Append CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 5. Add Position Embedding
        x = x + self.pos_embedding
        
        return x


class SimpleCNN(RecursiveNet):
    def __init__(self, in_channels=1, num_classes=10, input_size=(28, 28), recursive_mode=True):
        super().__init__(state_dim=512, recursive_mode=recursive_mode)
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_size)
            self.flattened_size = self.cnn_extractor(dummy).numel()

        self.input_embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.STATE_DIM),
            nn.ReLU(),
        )
        self.backbone = nn.Sequential(
            nn.Linear(self.STATE_DIM, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.STATE_DIM),
            nn.ReLU(),
        )
        self.output_head = nn.Linear(self.STATE_DIM, num_classes)

    def get_input_embedding(self, raw_input):
        x = self.cnn_extractor(raw_input)
        return self.input_embed(x)

class ResNet(RecursiveNet):
    def __init__(self, num_classes=10, pretrained=False, recursive_mode=True, state_dim=512, 
                 use_precomputed_features=False): 
        super().__init__(state_dim=state_dim, recursive_mode=recursive_mode)
        self.STATE_DIM = state_dim
        self.recursive_mode = recursive_mode
        self.pretrained = pretrained
        self.use_precomputed_features = use_precomputed_features

        if self.use_precomputed_features and not self.pretrained:
            raise ValueError("Cannot use_precomputed_features if model is not pretrained.")
            
        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.feature_extractor = nn.Sequential(*(list(resnet.children())[:-1]))

        if pretrained:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.feature_extractor.eval()

        self.input_embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, self.STATE_DIM),
            nn.ReLU(),
        )
        self.backbone = nn.Sequential(
            nn.Linear(self.STATE_DIM, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.STATE_DIM),
            nn.ReLU(),
        )
        self.output_head = nn.Linear(self.STATE_DIM, num_classes)
        self.resize_transform = transforms.Resize((224, 224)) if pretrained else nn.Identity()

    def train(self, mode=True):
        super().train(mode)
        if self.pretrained:
            self.feature_extractor.eval()
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
        return self

    def get_input_embedding(self, raw_input: torch.Tensor) -> torch.Tensor:
            if self.use_precomputed_features:
                return self.input_embed(raw_input)

            if raw_input.shape[1] == 1:
                raw_input = raw_input.repeat(1, 3, 1, 1)

            raw_input = self.resize_transform(raw_input)
            
            if self.pretrained:
                with torch.no_grad():
                    x = self.feature_extractor(raw_input)
            else:
                x = self.feature_extractor(raw_input)
            
            return self.input_embed(x)