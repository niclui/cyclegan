import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Code adapted from: github.com/aitorzip/PyTorch-CycleGAN/
# Let us define our discriminator and generator components
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        conv_block = [  nn.Conv2d(in_features, in_features, 3, padding=1, padding_mode='reflect'), # Convolutional block
                        nn.InstanceNorm2d(in_features), # Normalize across each input feature, across all the channels
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_features, in_features, 3, padding=1, padding_mode='reflect'),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block) # Make them into a sequential list!

    def forward(self, x):
        return x + self.conv_block(x) # Recall, in Res Block, you concat the input with the output. This deals with problem of vanishing gradients for early inputs because they get propagated forward

# Ok the objective of generator is to create synthetic image
class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9): # The paper uses 9 blocks for 256x256 images
        super().__init__()

        # Initial convolution block       
        model = [   nn.Conv2d(input_nc, 64, 7, padding=3, padding_mode='reflect'), # Essentially we are using 64 convolutional filters with reflective padding of 3
                    nn.InstanceNorm2d(64), # Do the instance norm stuff
                    nn.ReLU(inplace=True) ] # ReLU activation

        # Downsampling
        # Reduces height and width, while expanding # of channels
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), 
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        # Expands height and width while reducing number of channels. This keeps # of params roughly similar while expanding spatial resolution.
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.Conv2d(64, output_nc, 7, padding=3, padding_mode='reflect'),
                    nn.Tanh() ] # tanh activation

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Discriminator helps us to test between real and synthetic images
class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ] # Leaky ReLU to fight against vanishing gradients

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)] # Output is a matrix

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)  # This spits out a single logit by averaging across the entire tensor


# # Now define the cycleGAN network
# class cycleGAN(nn.Module):
#     def __init__(self, cuda, input_nc=3, output_nc=3, n_residual_blocks=9):
#         super().__init__()

#         # Now define my component models
#         self.G_A2B = Generator(input_nc, output_nc)
#         self.G_B2A = Generator(input_nc, output_nc)
#         self.D_A = Discriminator(input_nc)
#         self.D_B = Discriminator(input_nc)

#     def forward(self, A, B):

#         # Convert the real A/B images into synthetic B/A images
#         B2A = self.G_B2A(B)
#         A2B = self.G_A2B(A)

#         # Then force roundtrip consistency
#         B2A2B = self.G_A2B(B2A)
#         A2B2A = self.G_B2A(A2B)

#         # Separately, try to see if I can recover the exact same image by passing it into generator with the same output domain
#         A2A = self.G_B2A(A)
#         B2B = self.G_A2B(B)

#         return B2A, A2B, B2A2B, A2B2A, A2A, B2B










