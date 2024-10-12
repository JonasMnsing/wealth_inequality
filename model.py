import numpy as np

class population:

    def __init__(self, N_agents : int) -> None:
        
        self.wealth     = np.ones(N_agents)
        self.N_agents   = N_agents
        self.agents     = np.arange(N_agents)
        self.W          = np.sum(self.wealth)
        self.W_mean     = self.W / self.N_agents

    def transaction(self, gamma : float, dt : float, eta : float):

        agents_12   = np.random.choice(self.agents, 2, replace=False)
        wealth_12   = self.wealth[agents_12]

        dw          = np.sqrt(gamma*dt)*np.min(wealth_12)*eta
        eta_expeted = eta * np.sqrt(dt / gamma) * ()

    
