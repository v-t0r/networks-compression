import numpy as np
import tensorly as tl
import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.models.mobilenetv3 import InvertedResidual
from torchvision.models.efficientnet import FusedMBConv, MBConv
from torchvision.models.resnet import BasicBlock
import math
from tensorly.decomposition import partial_tucker

#Variavel global que define quais camadas são iteraveis
iterableTypes = [InvertedResidual, 
                    Conv2dNormActivation,
                    nn.modules.container.Sequential,
                    SqueezeExcitation,
                    FusedMBConv,
                    MBConv,
                    BasicBlock]

tl.set_backend('pytorch')

def isIterable(layer):
    global iterableTypes
    return type(layer) in iterableTypes

def decompose_network(network, rank = 0.5):
    decompose_block(network, rank)
    return network

def decompose_block(block, rank):
    for module in block._modules:
        layer = block._modules[module]
        if type(layer) is torch.nn.Conv2d:
            ranks = [math.ceil(layer.weight.size(0)*rank), math.ceil(layer.weight.size(1)*rank)]
            if layer.weight.size(1) != 1:
                print("Decompondo layer", module, "Dim. originais:", layer.weight.size(), "Dim. novas:", [ranks[1], ranks[0]])
                block._modules[module] = decompose_conv2d(layer, ranks)
            else:
                print("Pulando etapa de decomposição")
        elif isIterable(layer):
            decompose_block(layer, rank)

def decompose_conv2d(layer, ranks = 0.5):
    """Recebe uma camada convolucional e o valor dos ranks, retorna um objeto nn.Sequential 
    com a camada decomposta utilizando Tucker Decomposition"""
    
    print("ranks recebidos:", ranks)

    (core_factors), _ = partial_tucker(layer.weight.data, modes=[0, 1], rank=ranks, init='svd')
    (core, factors) = core_factors
    [last, first] = factors
    
    # Uma convolução que reduz os canais de S para R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0],             
                                  out_channels=first.shape[1], 
                                  kernel_size=1,
                                  stride=1, padding=0, 
                                  dilation=layer.dilation, 
                                  bias=False)

    # Uma convolução com R3 canais de entrada e R4 canais de saída
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1],             
                                 out_channels=core.shape[0], 
                                 kernel_size=layer.kernel_size,
                                 stride=layer.stride, 
                                 padding=layer.padding, 
                                 dilation=layer.dilation,
                                 bias=False)

    # Um convolução que aumenta os canais de R4 para T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], 
                                 out_channels=last.shape[0],
                                 kernel_size=1, stride=1,
                                 padding=0, dilation=layer.dilation,
                                 bias=True)

    first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1) #"deitando" o tensor, removendo dimensões inutilizadas e adiciona à camada.
    
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1) #removendo dimensões inutilizadas e e adiciona à camada.
    
    core_layer.weight.data = core #adicionando a camada

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers) #devolvendo o Sequential
 
def count_params_vgg(network):
    network = network.features
    return sum(np.prod(p.size()) for p in network.parameters())
    
def count_params_resnet(model):
    i = 0
    total_sum = 0
    for name, tensor in model.named_parameters():
        if ("conv" in name or "downsample" in name) and "weight" in name:
            print(i, ":", name)
            total_sum += np.prod(tensor.size())
        i += 1
    return total_sum

def count_params(model):
    """Retorna o número total de parâmetros da rede"""
    return sum(np.prod(p.size()) for p in model.parameters())