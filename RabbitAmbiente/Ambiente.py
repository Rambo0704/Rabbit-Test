import numpy as np

class Ambiente:
  def __init__(self):
    self.limite = (100,100) ## 100 por 100
    self.limite_passos = 500

  def reinicializar(self):
    self.robo = np.array([50,50]) ##inicializa no meio, array X/Y 
    self.predador = np.array([np.random.randint(0, self.limite[0]),np.random.randint(0, self.limite[1])])
    self.passos = 0
    return self.observar() ##Simula Sensores
  def observar(self):
    dist_frente = self.limite[1] - self.robo[1]
    dist_tras = self.robo[1]
    dist_dir = self.limite[0] - self.robo[0]
    dist_esq = self.robo[0]
    velocidade = 1
    distancia_predador = np.linalg.norm(self.robo - self.predador)

    return np.array([dist_frente, dist_tras, dist_dir, dist_esq, velocidade, distancia_predador],dtype=np.float32)
  def step(self, acao):
    ## 0 = frente, 1 = direita, 2 = esquerda, 3 = parar, 4 = trás
    #ação da presa
    if acao == 0:
        self.robo += np.array([0, 1])   
    elif acao == 1:
        self.robo += np.array([1, 0])    
    elif acao == 2:
        self.robo += np.array([-1, 0])  
    elif acao == 3:
        pass                            
    elif acao == 4:
        self.robo += np.array([0, -1]) 
    #açao do predador (que sempre vai na direçao da presa)
    direcao = self.robo - self.predador
    direcao = direcao / (np.linalg.norm(direcao) + 1e-6)
    self.predador += direcao.astype(int)

    self.passos += 1
    obs = self.observar()
    recompensa, done = self.recompensa(obs)

    return obs, recompensa, done   
  def recompensa(self, obs):
    dist_frente, dist_tras, dist_dir, dist_esq, _, distancia_predador = obs

    if (dist_frente <= 0) or (dist_tras <= 0) or (dist_dir <= 0) or (dist_esq <= 0):
      return -20, True  # Bateu

    if distancia_predador < 2:
            return -50, True  # Foi pego

    if self.passos >= self.limite_passos:
      return +100, True  # Fugiu com sucesso

    return +1, False  # Está fugindo 