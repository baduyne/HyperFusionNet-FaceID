import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
import numpy as np
import cv2
import pytorch_msssim

# Hàm mất mát SSIM (Structural Similarity Index Measure)
# Khởi tạo ssim_fn với data_range và channel=1 cho ảnh grayscale
ssim_fn = pytorch_msssim.SSIM(data_range=1.0, channel=1)

def pixel_loss(pred, target):
    """
    Calculates the pixel-wise Mean Squared Error loss.
    """
    return F.mse_loss(pred, target)

def gradient_loss(img):
    """
    Calculates the gradient loss (Facial Detail Preserving Loss - non-reference).
    This encourages smoother reconstructed images by penalizing large intensity changes.
    """
    # Calculate gradients along width (dx)
    # dx = |img(x,y+1) - img(x,y)|
    dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    # Calculate gradients along height (dy)
    # dy = |img(x+1,y) - img(x,y)|
    dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    return torch.mean(dx) + torch.mean(dy)

def ssim_loss(pred, target):
    """
    Calculates the SSIM loss (1 - SSIM).
    The ssim_fn object is already configured with data_range=1.0 and channel=1.
    """
    return 1 - ssim_fn(pred, target)

def preprocess_image(img_path, size=(128, 128)):
    """
    Preprocesses an image: reads as grayscale, resizes, normalizes to [0, 1],
    and converts to a PyTorch tensor of shape [1, 1, H, W].
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return img

# ===== Dense Block =====
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        # Concatenate input with the output of Conv-ReLU along the channel dimension
        return torch.cat([x, out], 1)

# ===== BSRDB Block (Bi-Scope Residual Dense Block) =====
class BSRDB(nn.Module):
    def __init__(self, in_channels=16, growth_rate=16):
        super().__init__()
        # Three DenseBlocks as part of the residual dense block
        self.db1 = DenseBlock(in_channels, growth_rate)
        self.db2 = DenseBlock(in_channels + growth_rate, growth_rate)
        self.db3 = DenseBlock(in_channels + 2 * growth_rate, growth_rate)
        
        # Final convolutional layer to map dense features to desired output channels (64)
        self.conv = nn.Conv2d(in_channels + 3 * growth_rate, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # 1x1 projection layer for the global residual connection, to match input channels to output (64)
        self.proj = nn.Conv2d(in_channels, 64, kernel_size=1)

    def forward(self, x):
        identity = x # Store original input for global residual connection
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        out = self.relu(self.conv(out))
        
        # Apply global residual connection
        identity_proj = self.proj(identity)
        return out + identity_proj

# ===== Encoder =====
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # BSRDB takes 16 input channels and outputs 64 channels
        self.bsrdb = BSRDB(16, 16) 

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.bsrdb(x) # Encoder output has 64 channels

# ===== Feedback Block =====
class FeedbackBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=3, padding=1) 

    def forward(self, prev_out): # prev_out here is the 1-channel final output from a previous decoder iteration
        return self.conv(prev_out)

# ===== Decoder =====
class Decoder(nn.Module):
    def __init__(self, T=4):
        super().__init__()
        self.T = T # Number of recurrent iterations
        
        # Convolutional layers of the decoder.
        # The input to the first layer is the concatenation of features from the Encoder (64 channels)
        # and the 'prev' feedback features (also 64 channels) => total 128 channels.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True), # Adjusted input channels to 128
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1) # Final layer outputs 1 channel for the reconstructed image
        )
        self.fb = FeedbackBlock() # Feedback block

    def forward(self, x): # x is the fused feature from the Encoder (64 channels
        B, _, H, W = x.shape
        prev = torch.zeros((B, 64, H, W), device=x.device) 
        
        for i in range(self.T):
            inp = torch.cat([x, prev], dim=1) 
            out = self.conv_layers(inp) # 'out' is the current output of the decoder (1 channel)
            
            # Update 'prev' for the next iteration via the FeedbackBlock.
            # Only update if it's not the last iteration.
            if i < self.T - 1: 
                prev = self.fb(out) 
            
        return out # Return the final reconstructed face image (1 channel)

# ===== Transfer Layer =====
class TransferLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(1) # Batch normalization for 1-channel images
        # Resize the image to 128x128 using bilinear interpolation
        self.resize = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.norm(x)
        return self.resize(x)

# ===== Full HyperFace Pipeline (without direct logits output) =====
class HyperFacePipeline(nn.Module):
    def __init__(self, num_classes=113): # num_classes is kept for ClassifierHead if needed, but not used in forward
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.transfer = TransferLayer()
        # Use FaceNet pretrained on VGGFace2
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()

        for param in self.facenet.parameters():
            param.requires_grad = False

    def forward(self, ir, vis):
        # 1. Pre-fusion scheme
        a1, a2 = 0.8, 0.2
        ir_mix = a1 * ir + a2 * vis
        vis_mix = a2 * ir + a1 * vis

        # 2. Siamese Encoder
        f_ir = self.encoder(ir_mix) # Features from mixed IR (64 channels)
        f_vis = self.encoder(vis_mix) # Features from mixed VIS (64 channels)
        
        # 3. Summation-based Feature Fusion strategy
        fused = f_ir + f_vis # Fused features (64 channels)

        # 4. Feedback-style Decoder to reconstruct the face
        fused_face = self.decoder(fused) # Reconstructed face image (1 channel)

        # 5. Transfer Layer
        transferred = self.transfer(fused_face) # 1-channel, 128x128 image

        # 6. Prepare for FaceNet (requires 3 RGB channels)
        fused_rgb = transferred.repeat(1, 3, 1, 1) # Repeat channel to create 3-channel image

        embeddings = self.facenet(fused_rgb) # 512-dimensional embedding vector from FaceNet

        return embeddings, fused_face # Return embeddings and fused_face
