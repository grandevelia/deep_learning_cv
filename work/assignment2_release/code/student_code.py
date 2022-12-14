from unittest.mock import patch
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
from custom_blocks import Mlp


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
        assert weight.size(2) == weight.size(3)
        assert input_feats.size(1) == weight.size(1)
        assert isinstance(stride, int) and (stride > 0)
        assert isinstance(padding, int) and (padding >= 0)

        # save the conv params
        kernel_size = weight.size(2)
        ctx.stride = stride
        ctx.padding = padding
        ctx.input_height = input_feats.size(2)
        ctx.input_width = input_feats.size(3)

        # make sure this is a valid convolution
        assert kernel_size <= (input_feats.size(2) + 2 * padding)
        assert kernel_size <= (input_feats.size(3) + 2 * padding)

        #################################################################################
        # Fill in the code here
        #################################################################################
        ctx.n_filters = weight.size(1)
        ctx.out_size = weight.size(0)
        ctx.bs = input_feats.size(0)
        ctx.kernel_size = kernel_size

        weight_blocks = unfold(weight, stride=kernel_size, kernel_size=kernel_size).squeeze(-1).unsqueeze(0)
        input_blocks = unfold(input_feats, stride=stride, padding=padding, kernel_size=kernel_size)

        output_height = int(((ctx.input_height + 2 * padding - weight.size(2))/stride) + 1)
        output_width = int(((ctx.input_width + 2 * padding - weight.size(2))/stride) + 1)

        output_size = (output_height, output_width)
        output = fold(weight_blocks@input_blocks, kernel_size=1, 
                        output_size=output_size, stride=1) + bias[None, :, None, None]

        # save for backward (you need to save the unfolded tensor into ctx)
        ctx.save_for_backward(weight_blocks, input_blocks, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward propagation of convolution operation

        Args:
          grad_output: gradients of the outputs

        Outputs:
          grad_input: gradients of the input features
          grad_weight: gradients of the convolution weight
          grad_bias: gradients of the bias term 

        """
        # unpack tensors and initialize the grads
        weight_blocks, input_blocks, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # recover the conv params
        kernel_size = ctx.kernel_size
        out_size = ctx.out_size
        bs = ctx.bs
        n_filters = ctx.n_filters
        input_height = ctx.input_height
        input_width = ctx.input_width
        stride = ctx.stride
        padding = ctx.padding

        #################################################################################
        # Fill in the code here
        #################################################################################
        # compute the gradients w.r.t. input and params
        output_blocks = unfold(grad_output, stride=1, kernel_size=1)
        grad_weight = (output_blocks @ input_blocks.permute([0, 2, 1])).view([bs, out_size, 
                        n_filters, kernel_size, kernel_size])
        output_size = (input_height, input_width)
        grad_input = fold((weight_blocks.permute([0, 2, 1]) @ output_blocks), kernel_size=kernel_size, 
                            stride=stride, padding=padding, output_size=output_size)
        if bias is not None and ctx.needs_input_grad[2]:
            # compute the gradients w.r.t. bias (if any)
            grad_bias = grad_output.sum((0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None


custom_conv2d = CustomConv2DFunction.apply


class CustomConv2d(Module):
    """
    The same interface as torch.nn.Conv2D
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CustomConv2d, self).__init__()
        assert isinstance(kernel_size, int), "We only support squared filters"
        assert isinstance(stride, int), "We only support equal stride"
        assert isinstance(padding, int), "We only support equal padding"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # not used (for compatibility)
        self.dilation = dilation
        self.groups = groups

        # register weight and bias as parameters
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # initialization using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # call our custom conv2d op
        return custom_conv2d(input, self.weight, self.bias, self.stride, self.padding)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, padding={padding}"
        )
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


class SimpleNet(nn.Module):
    # a simple CNN for image classifcation
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(SimpleNet, self).__init__()
        # you can start from here and create a better model
        self.features = nn.Sequential(
            # conv1 block: conv 7x7
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv2 block: simple bottleneck
            conv_op(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(64, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv3 block: simple bottleneck
            conv_op(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(128, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.attack = default_attack(loss_fn=None, num_steps=3)

    def reset_parameters(self):
        # init all params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.consintat_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # you can implement adversarial training here
        if self.training:
            self.eval()
            randoms = np.random.randint(x.shape[0], size=x.shape[0]//2)
            perts = x[randoms, :, :, :]
            with torch.set_grad_enabled(True):
                adv_x = self.attack.perturb(self, perts)
            x[randoms] = adv_x#torch.concat((x, adv_x))
            self.train()
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, conv_op, in_chans, intmd_chans, out_chans):
        super().__init__()
        self.conv1 = conv_op(in_chans, intmd_chans, kernel_size=1, stride=1, padding=0)
        self.norm1 = nn.BatchNorm2d(intmd_chans)
        self.conv2 = conv_op(intmd_chans, intmd_chans, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(intmd_chans)
        self.conv3 = conv_op(intmd_chans, out_chans, kernel_size=1, stride=1, padding=0)
        self.norm3 = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)
        self.input_expander = conv_op(in_chans, out_chans, kernel_size=1, stride=1)
        

    def forward(self, x):
        orig_x = x.clone()
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        orig_x = self.input_expander(orig_x)
        x = x + orig_x
        x = self.relu(x)
        return x

#################################################################################
# Part II: Design and train a network
#################################################################################
class MyNet(nn.Module):
    # a simple CNN for image classifcation
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(MyNet, self).__init__()
        # you can start from here and create a better model
        self.features = nn.Sequential(
            # conv1 block: conv 7x7
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv2 block: simple bottleneck
            BottleneckBlock(conv_op, 64, 64, 128),
            BottleneckBlock(conv_op, 128, 128, 128),
            BottleneckBlock(conv_op, 128, 128, 128),

            #TOOD Another max pool here ?
            # conv3 block: simple bottleneck
            BottleneckBlock(conv_op, 128, 256, 256),
            BottleneckBlock(conv_op, 256, 256, 256),
            BottleneckBlock(conv_op, 256, 256, 256),
            
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv4 block: simple bottleneck
            BottleneckBlock(conv_op, 256, 128, 512),
        )
        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def reset_parameters(self):
        # init all params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.consintat_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # you can implement adversarial training here
        # if self.training:
        #   # generate adversarial sample based on x
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=128,
        num_classes=100,
        patch_size=16,
        in_chans=3,
        embed_dim=192,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        window_size=4,
        window_block_indexes=(1, 3),
    ):
        """
        Args:
            img_size (int): Input image size.
            num_classes (int): Number of object categories
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            E.g., [0, 2] indicates the first and the third blocks will use window attention.

        Feel free to modify the default parameters here.
        """
        super(SimpleViT, self).__init__()
        if use_abs_pos:
            # Initialize absolute positional embedding with image size
            # The embedding is learned from data
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        ########################################################################
        # Fill in the code here
        ########################################################################
        # the implementation shall start from embedding patches,
        # followed by some transformer blocks
        self.patch_emb = PatchEmbed(
            (patch_size, patch_size), 
            (patch_size, patch_size),
            padding=(0, 0),
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        transformer_blocks = [
            TransformerBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop_path=dpr[i],
                norm_layer=norm_layer, 
                act_layer=act_layer, 
                window_size=window_size if i in window_block_indexes else 0
            )
            for i, _ in enumerate(range(depth))
        ]
        self.transformer_blocks = nn.Sequential(
            *transformer_blocks
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)
        # add any necessary weight initialization here

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        ########################################################################
        # Fill in the code here
        ########################################################################
        x = self.patch_emb(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for layer in self.transformer_blocks:
            x = layer(x)

        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = x.mean(dim = 1) #Average pool
        x = self.head(x)
        return x


# change this to your model!
default_cnn_model = SimpleNet
default_vit_model = SimpleViT

# define data augmentation used for training, you can tweak things if you want
def get_train_transforms(normalize):
    train_transforms = []
    train_transforms.append(transforms.Scale(144))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomColor(0.15))
    train_transforms.append(transforms.RandomRotate(15))
    train_transforms.append(transforms.RandomSizedCrop(128))
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(normalize)
    train_transforms = transforms.Compose(train_transforms)
    return train_transforms


# define data augmentation used for validation, you can tweak things if you want
def get_val_transforms(normalize):
    val_transforms = []
    val_transforms.append(transforms.Scale(144))
    val_transforms.append(transforms.CenterCrop(128))
    val_transforms.append(transforms.ToTensor())
    val_transforms.append(normalize)
    val_transforms = transforms.Compose(val_transforms)
    return val_transforms


#################################################################################
# Part III: Adversarial samples and Attention
#################################################################################
class PGDAttack(object):
    def __init__(self, loss_fn, num_steps=10, step_size=0.01, epsilon=0.1):
        """
        Attack a network by Project Gradient Descent. The attacker performs
        k steps of gradient descent of step size a, while always staying
        within the range of epsilon (under l infinity norm) from the input image.

        Args:
          loss_fn: loss function used for the attack
          num_steps: (int) number of steps for PGD
          step_size: (float) step size of PGD (i.e., alpha in our lecture)
          epsilon: (float) the range of acceptable samples
                   for our normalization, 0.1 ~ 6 pixel levels
        """
        self.loss_fn = loss_fn
        self.num_steps = num_steps
        self.step_size = step_size
        self.epsilon = epsilon

    def perturb(self, model, input):
        """
        Given input image X (torch tensor), return an adversarial sample
        (torch tensor) using PGD of the least confident label.

        See https://openreview.net/pdf?id=rJzIBfZAb

        Args:
          model: (nn.module) network to attack
          input: (torch tensor) input image of size N * C * H * W

        Outputs:
          output: (torch tensor) an adversarial sample of the given network
        """
        # clone the input tensor and disable the gradients
        output = input.clone()
        input.requires_grad = False
        # loop over the number of steps
        # for _ in range(self.num_steps):
        #################################################################################
        # Fill in the code here
        #################################################################################
        output = output.detach()
        for _ in range(self.num_steps):
            output_nograd = output.clone().detach()
            output_nograd.requires_grad = True
            preds = model(output_nograd)
            min_preds, min_inds = torch.min(preds, 1)
            min_preds.backward(torch.ones_like(min_preds))
            with torch.no_grad():
                delta_gradient = self.step_size * output_nograd.grad.sign()
                output_nograd -= delta_gradient
            
            output = torch.min(torch.max(input - self.epsilon, output_nograd), input + self.epsilon)
            model.zero_grad()
        input.requires_grad = True
        return output


default_attack = PGDAttack


class GradAttention(object):
    def __init__(self, loss_fn):
        """
        Visualize a network's decision using gradients

        Args:
          loss_fn: loss function used for the attack
        """
        self.loss_fn = loss_fn

    def explain(self, model, input):
        """
        Given input image X (torch tensor), return a saliency map
        (torch tensor) by computing the max of abs values of the gradients
        given by the predicted label

        See https://arxiv.org/pdf/1312.6034.pdf

        Args:
          model: (nn.module) network to attack
          input: (torch tensor) input image of size N * C * H * W

        Outputs:
          output: (torch tensor) a saliency map of size N * 1 * H * W
        """
        # make sure input receive grads
        input.requires_grad = True
        if input.grad is not None:
            input.grad.zero_()
        
        #################################################################################
        # Fill in the code here
        #################################################################################
        preds = model(input)
        max_preds, inds = torch.max(preds, 1)
        max_preds.backward(torch.ones_like(max_preds))
        output, chan_inds = torch.max(torch.abs(input.grad), 1)
        return output.unsqueeze(1)


default_attention = GradAttention


def vis_grad_attention(input, vis_alpha=2.0, n_rows=10, vis_output=None):
    """
    Given input image X (torch tensor) and a saliency map
    (torch tensor), compose the visualziations

    Args:
      input: (torch tensor) input image of size N * C * H * W
      output: (torch tensor) input map of size N * 1 * H * W

    Outputs:
      output: (torch tensor) visualizations of size 3 * HH * WW
    """
    # concat all images into a big picture
    input_imgs = make_grid(input.cpu(), nrow=n_rows, normalize=True)
    if vis_output is not None:
        output_maps = make_grid(vis_output.cpu(), nrow=n_rows, normalize=True)

        # somewhat awkward in PyTorch
        # add attention to R channel
        mask = torch.zeros_like(output_maps[0, :, :]) + 0.5
        mask = output_maps[0, :, :] > vis_alpha * output_maps[0, :, :].mean()
        mask = mask.float()
        input_imgs[0, :, :] = torch.max(input_imgs[0, :, :], mask)
    output = input_imgs
    return output


default_visfunction = vis_grad_attention
