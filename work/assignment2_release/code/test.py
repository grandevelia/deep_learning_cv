import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.functional import fold, unfold
from torchvision.utils import make_grid
import math

from utils import resize_image
import custom_transforms as transforms
from custom_blocks import PatchEmbed, TransformerBlock, trunc_normal_


#################################################################################
# You will need to fill in the missing code in this file
#################################################################################


#################################################################################
# Part I: Understanding Convolutions
#################################################################################
class CustomConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input_feats, weight, bias, stride=1, padding=0):
        """
        Forward propagation of convolution operation.
        We only consider square filters with equal stride/padding in width and height!

        Args:
          input_feats: input feature map of size N * C_i * H * W
          weight: filter weight of size C_o * C_i * K * K
          bias: (optional) filter bias of size C_o
          stride: (int, optional) stride for the convolution. Default: 1
          padding: (int, optional) Zero-padding added to both sides of the input. Default: 0

        Outputs:
          output: responses of the convolution  w*x+b

        """
        # sanity check

#In[1]
out_size = 5
n_filters = 3
kernel_size = 8
h = 100
w = 100
bs = 2
weight = torch.tensor(np.random.random(size=[out_size, n_filters, kernel_size, kernel_size]))
bias = torch.tensor(np.random.random(size=[out_size]))
input_feats = torch.tensor(np.random.random(size=[bs, n_filters, h, w]))
stride = 1
padding = 1
#In[1]

weight_blocks = unfold(weight, stride=kernel_size, kernel_size=kernel_size).squeeze(-1).unsqueeze(0)
input_blocks = unfold(input_feats, stride=stride, padding=padding, kernel_size=kernel_size)

nc = input_feats.size(0)
ni = input_feats.size(1)
output_height = int(((input_feats.size(2) + 2 * padding - weight.size(2))/stride) + 1)
output_width = int(((input_feats.size(3) + 2 * padding - weight.size(2))/stride) + 1)
output_size = (output_height, output_width)
output = fold(weight_blocks@input_blocks, kernel_size=1, 
                output_size=output_size, stride=1) + bias[None, :, None, None]

#In[2]
# save for backward (you need to save the unfolded tensor into ctx)
ctx.save_for_backward(weight_blocks, input_blocks, bias)



   
# grad_output: gradients of the outputs

# Outputs:
# grad_input: gradients of the input features
# grad_weight: gradients of the convolution weight
# grad_bias: gradients of the bias term

# unpack tensors and initialize the grads
# your_vars, weight, bias = ctx.saved_tensors
grad_input = grad_weight = grad_bias = None

weight_blocks, input_blocks, bias = ctx.saved_tensors

# recover the conv params
kernel_size = weight.size(2)

#################################################################################
# Fill in the code here
#################################################################################
# compute the gradients w.r.t. input and params
output_blocks = unfold(grad_output, stride=1, kernel_size=1)
grad_weight = (output_blocks @ input_blocks.permute([0, 2, 1])).view([bs, out_size, 
                ctx.n_filters, ctx.kernel_size, ctx.kernel_size])
output_size = (ctx.input_height, ctx.input_width)
grad_input = fold((weight_blocks.permute([0, 2, 1]) @ output_blocks), kernel_size=ctx.kernel_size, 
                    stride=ctx.stride, padding=ctx.padding, output_size=output_size)

if bias is not None and ctx.needs_input_grad[2]:
# compute the gradients w.r.t. bias (if any)
grad_bias = grad_output.sum((0, 2, 3))

return grad_input, grad_weight, grad_bias, None, None

