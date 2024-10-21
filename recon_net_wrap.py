import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from fastmri.models import Unet
from recon_net import ReconNet

def initialize_conv1d_as_delta(conv1,channels):
    # Ensure that this operation is not tracked by autograd
    torch.nn.init.zeros_(conv1.weight)
    #torch.nn.init.zeros_(conv1.bias)
    # set identity kernel
    #print (conv1.weight.data.shape)
    for i in range(channels):
        conv1.weight.data[i, i, 1] = torch.ones((1,1))



def initialize_conv2d_as_delta(conv2,channels):
    # Get the weight tensor of the convolutional layer
    with torch.no_grad():
        conv2.weight[:, :, :, :] = 0.0
        conv2.bias[:] = 0.0
        for i in range(channels):
            conv2.weight[i, i, 2, 2] = 1.0 # our equivalent delta-dirac

def initialize_conv2d_as_delta_noBias(conv2,channels):
    # Get the weight tensor of the convolutional layer
    with torch.no_grad():
        conv2.weight[:, :, :, :] = 0.0
        for i in range(channels):
            conv2.weight[i, i, 3, 3] = 1.0 # our equivalent delta-dirac


class ViTfuser(nn.Module):
    def __init__(self, net, epsilon=1e-6):
        super().__init__()
        self.device = 'cuda:0'

        # ViT layer
        self.recon_net = ReconNet(net).to(self.device)#.requires_grad_(False)
        self.recon_net_ref = ReconNet(net).to(self.device)
        # Load weights - Natural or MRI start
        
        cp = torch.load('./lsdir-2x+hq50k_vit_epoch_60.pt', map_location=self.device) # Natural images training
        self.recon_net.load_state_dict(cp['model_state_dict'])
        self.recon_net_ref.load_state_dict(cp['model_state_dict'])
        """
        cp = torch.load('./L2_checkpoints_ViT_only/model_400.pt', map_location=self.device)
        self.recon_net.load_state_dict(cp['model'])
        self.recon_net_ref.load_state_dict(cp['model'])
        """
        # Fusion layers
        self.epsilon = epsilon

        self.param1 = nn.Parameter(torch.normal(1, 0.01, size=(198,)))
        self.param2 = nn.Parameter(torch.normal(0, 0.01, size=(198,)))



    def printer(self, x):
        print("Current value of param1 during forward:", self.param1)
        return

    def forward(self, img,ref): #,ref
        # Norm
        #print("Current value of param1 during forward:", self.param1)
        #print("Current value of param2 during forward:", self.param2)
        in_pad, wpad, hpad = self.recon_net.pad(img)
        ref_pad, wpad, hpad = self.recon_net.pad(ref)
        input_norm,mean,std = self.recon_net.norm(in_pad.float())
        ref_norm,mean_ref,std_ref = self.recon_net.norm(ref_pad.float())
        #print("Weights of the Conv1 layer:")
        #print(self.conv1.weight)        
        # Feature extract
        features = self.recon_net.net.forward_features(input_norm)#.permute(0,2,1)
        #features = self.conv1_acq1(features)
        
        
        features_ref = self.recon_net_ref.net.forward_features(ref_norm)
        # Fusion
        
        batch_size, num_channels, height = features.shape
        features_flat = features.reshape(batch_size, num_channels, -1)
        features_ref_flat = features_ref.reshape(batch_size, num_channels, -1)       
        
        # Reshape params to match the dimensions
        param1_expanded = self.param1.reshape(1, -1, 1)  # Shape: [1, 416, 1]
        param2_expanded = self.param2.reshape(1, -1, 1)  # Shape: [1, 416, 1]
        # Expand params to match the flattened tensor dimensions
        param1_expanded = param1_expanded.expand(batch_size, -1, height)  # Shape: [batch_size, 416, height*width]
        param2_expanded = param2_expanded.expand(batch_size, -1, height)  # Shape: [batch_size, 416, height*width]
        # Calculate weighted sum

        weighted_sum = (param1_expanded * features_flat + param2_expanded * features_ref_flat)

        
        # Calculate normalization factor
        normalization_factor = param1_expanded + param2_expanded + self.epsilon
        
        # Normalize
        features_comb = weighted_sum / normalization_factor
        
        # Low Resolution
        features_comb = features_comb.reshape(features_flat.shape[0], 198, 1024)
        
        # Recon Head
        head_out = self.recon_net.net.head(features_comb)
        
        # Low Resolution 
        head_out_img = self.recon_net.net.seq2img(head_out, (180, 110))


        # un-norm
        merged = self.recon_net.unnorm(head_out_img, mean, std) 

        # un-pad 
        im_out = self.recon_net.unpad(merged,wpad,hpad)
        
        return im_out