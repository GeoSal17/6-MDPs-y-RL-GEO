"""
El camión mágico, pero ahora por simulación

"""

from RL import MDPsim, SARSA, Q_learning
from random import random, randint

class CamionMagico(MDPsim):
    """
    Clase que representa un MDP para el problema del camión mágico.
    
    Si caminas, avanzas 1 con coso 1
    Si usas el camion, con probabilidad rho avanzas el doble de donde estabas
    y con probabilidad 1-rho te quedas en el mismo lugar. Todo con costo 2.
    
    El objetivo es llegar a la meta en el menor costo posible
    
    """    
    
    def __init__(self, gama, rho, meta):
        self.gama = gama
        self.rho = rho
        self.meta = meta
        self.estados = tuple(range(1, meta + 2))
    
    def estado_inicial(self):
        #return randint(1, self.meta // 2 + 1)
        return randint(1, self.meta - 1)
    
    def acciones_legales(self, s):
        if s >= self.meta:
            return []
        return ['caminar', 'usar_camion']
    
    def recompensa(self, s, a, s_):
        return (
            -100  if s_ > self.meta else
             100  if s_ == self.meta else
            -1  if a == 'caminar' else -2   
        ) 
        
    def transicion(self, s, a):
        if a == 'caminar':
            return min(s + 1, self.meta + 1)
        elif a == 'usar_camion':
            return min(self.meta + 1, 2*s) if random() < self.rho else s
        
    def es_terminal(self, s):
        return s >= self.meta

mdp_sim = CamionMagico(
    gama=0.999, rho=0.9, meta=145
)
    
Q_sarsa = SARSA(
    mdp_sim, 
    alfa=0.1, epsilon=0.02, n_ep=100_000, n_iter=50
)
pi_s = {s: max(
    ['caminar', 'usar_camion'], key=lambda a: Q_sarsa[(s, a)]
) for s in mdp_sim.estados if not mdp_sim.es_terminal(s)}

Q_ql = Q_learning(
    mdp_sim, 
    alfa=0.1, epsilon=0.02, n_ep=100_000, n_iter=1000
)
pi_ql = {s: max(
    ['caminar', 'usar_camion'], key=lambda a: Q_ql[(s, a)]
) for s in mdp_sim.estados if not mdp_sim.es_terminal(s)}

print(f"Los tramos donde se debe usar el camión segun SARSA son:")
print([s for s in pi_s if pi_s[s] == 'usar_camion'])
print("-"*50)
print(f"Los tramos donde se debe usar el camión segun Qlearning son:")
print([s for s in pi_ql if pi_ql[s] == 'usar_camion'])
print("-"*50)


"""
**********************************************************************************
Ahora responde a las siguientes preguntas:
**********************************************************************************

- Prueba con diferentes valores de rho. ¿Qué observas? ¿Porqué crees que pase eso?
    Al aumentar rho, se usa más el camión, supongo que porque suele ser más rentable
    que caminar, pero cuando se disminuye, utiliza menos el camión, ya que es más 
    costoso.

- Prueba con diferentes valores de gama. ¿Qué observas? ¿Porqué crees que pase eso?
    Con gama bajo, no se enfoca tanto en llegar a la meto, sino en la recompensas
    inmediatas, pasa lo contrario con gama alto.

- ¿Qué tan diferente es la política óptima de SARSA y Q-learning?
    Q-learning es más ambicioso que SARSA, supongo que SARSA va a medir más las
    acciones que Q-learning.

- ¿Cambia mucho el resultado cambiando los valores de recompensa?
    Si, ya que el algoritmo se basa en minimizar o maximizar la suma de recompensas,
    si se penalizara más el usar el camión, al final solo caminaríamos más.

- ¿Cuantas iteraciones se necesitan para que funcionen correctamente los algoritmos?
    Yo digo que unas 100,000 está bien para garantizar una política estable.

- ¿Qué pasaria si ahora el estado inicial es cualquier estado de la mitad para abajo?
    El aprendizaje se enfocaría más en esa mitad y aprendería menos de la mitad para
    arriba, haciendo que los valores de la segunda mitad sean imprecisos o inestables.
**********************************************************************************

"""