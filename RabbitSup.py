import torch 
import torch.nn as nn
import numpy as np
import torch.optim as optim

##np.seed(42) se eu quiser que se mantenha as amostras

X = np.random.uniform(0,100(100,5)) ##por agora vou criar diferentes dados para treinamento
#definir uma regra
y = ((X[:, 0] < 30) & (X[:, 2] < 30)).astype(int) ## isso que deixa ele apredizado supervisionado

X_tensor = torch.tensor(X,dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

class RabbitNet(nn.Module):
    def __init__(self):
        super(RabbitNet, self).__init__()
        self.rede = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid() 
        )
    def forward(self, x):
        return self.rede(x)

modelo = RabbitNet

criterio = nn.BCELoss()
otimizador = optim.Adam(modelo.parameters(),lr = 0.1)

for epoca in range(500):
    saida = modelo(X_tensor)
    perda = criterio(saida, y_tensor)

    otimizador.zero_grad()
    perda.backward()
    otimizador.step()

    if epoca % 50 == 0:
        print(f'Epoca {epoca} - Perda: {perda.item():.4f}')