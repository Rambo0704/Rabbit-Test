import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from Ambiente import Ambiente
import matplotlib.pyplot as plt
#rede neural
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.rede = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.rede(x)
#memoria
class ReplayMemory:
    def __init__(self, capacidade):
        self.memoria = deque(maxlen=capacidade)

    def armazenar(self, transicao):
        self.memoria.append(transicao)

    def sample(self, tamanho):
        return random.sample(self.memoria, tamanho)

    def __len__(self):
        return len(self.memoria)
#Parametros
TAMANHO_ESTADO = 6
N_ACOES = 5
MEMORIA = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.001
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
EPISODIOS = 500

env = Ambiente()
q_network = QNetwork(TAMANHO_ESTADO, N_ACOES)
target_network = QNetwork(TAMANHO_ESTADO, N_ACOES)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=LR)
memoria = ReplayMemory(MEMORIA)
criterio = nn.MSELoss()

for episodio in range(EPISODIOS):
    estado = env.reinicializar()
    estado = torch.tensor(estado, dtype=torch.float32)
    total_recompensa = 0

    done = False

    while not done:
        # Epsilon-Greedy
        if random.random() < EPSILON:
            acao = random.randint(0, N_ACOES - 1)
        else:
            with torch.no_grad():
                q_vals = q_network(estado)
                acao = q_vals.argmax().item()

        proximo_estado, recompensa, done = env.step(acao)
        proximo_estado = torch.tensor(proximo_estado, dtype=torch.float32)

        memoria.armazenar((estado, acao, recompensa, proximo_estado, done))

        estado = proximo_estado
        total_recompensa += recompensa

        # Treinamento
        if len(memoria) >= BATCH_SIZE:
            batch = memoria.sample(BATCH_SIZE)
            estados, acoes, recompensas, proximos_estados, finais = zip(*batch)

            estados = torch.stack(estados)
            acoes = torch.tensor(acoes)
            recompensas = torch.tensor(recompensas, dtype=torch.float32)
            proximos_estados = torch.stack(proximos_estados)
            finais = torch.tensor(finais, dtype=torch.bool)

            q_atual = q_network(estados).gather(1, acoes.unsqueeze(1)).squeeze()

            q_proximo = target_network(proximos_estados).max(1)[0]
            q_proximo[finais] = 0.0

            q_alvo = recompensas + GAMMA * q_proximo

            perda = criterio(q_atual, q_alvo.detach())

            optimizer.zero_grad()
            perda.backward()
            optimizer.step()

    # Atualizar rede alvo
    if episodio % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())

    # Decair epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY
    if episodio % 50 == 0:
      print(f"Episódio {episodio}, Recompensa Total: {total_recompensa}, Epsilon: {EPSILON:.3f}")
