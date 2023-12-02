import torch
from torchvision import transforms, datasets #carregar dataset
from torch.utils.data import DataLoader #dataloader
import torch.nn as nn #função de erro
import torch.optim as optim #otimizador
import time #medir o tempo
import sys #pegar parametros

import nn_decomposer as nnd #decompor

device = torch.device('cpu')

param = sys.argv[1:]
end = ""
if len(param) == 1:
    end = param[0]
else:
    print("This program expects one parameter: the network path.")
    exit(1)
    
model = torch.load(end, map_location = device)

print(model)

params_before = nnd.count_params(model)

# chamando a função que decompõe a rede
nnd.decompose_network(model, rank = 0.07)

params_after = nnd.count_params(model)

print('Numero de parametros antes da decomposição: {params_before}'.format(params_before = params_before))
print('Numero de parametros depois da decomposição: {params_after}'.format(params_after = params_after))
print('Ratio: {ratio}'.format(ratio = float(params_after)/params_before))

torch.save(model, './decomposed_model.pt')
