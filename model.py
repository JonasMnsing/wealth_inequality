import numpy as np

class economy:

    def __init__(self, N_agents : int) -> None:
        
        self.wealth     = np.ones(N_agents)
        self.N_agents   = N_agents
        self.agents     = np.arange(N_agents)
        self.W          = np.sum(self.wealth)
        self.W_mean     = self.W / self.N_agents

    def transaction(self, gamma : float, dt : float, chi=0.0):

        agents  = np.random.choice(self.agents, 2, replace=False)
        w1      = self.wealth[agents[0]]
        w2      = self.wealth[agents[1]]

        beta    = np.sqrt(gamma*dt)
        dw      = beta*np.min([w1,w2])

        if np.random.rand() > 0.5:

            self.wealth[agents[0]]  += dw
            self.wealth[agents[1]]  -= dw
        
        else:

            self.wealth[agents[0]]  -= dw
            self.wealth[agents[1]]  += dw

    def run_economy(self, N_transactions : float, dt : float, gamma : float, chi=0.0) -> np.array:
        
        wealth_storage      = np.zeros(shape=(N_transactions,self.N_agents))
        wealth_storage[0,:] = self.wealth

        for t in range(1, N_transactions):

            self.transaction(gamma=gamma, dt=dt, chi=chi)
            wealth_storage[t,:] = self.wealth

        return wealth_storage

    def return_wealth_distribution(self):

        return self.wealth
    
if __name__ == '__main__':

    # Parameter
    N_agents        = 100
    N_transactions  = 1000
    gamma           = 0.5
    chi             = 0.0
    dt              = 1.0

    system          = economy(N_agents=N_agents)
    wealth_storage  = system.run_economy(N_transactions=N_transactions, dt=dt, gamma=gamma, chi=chi)

    np.savetxt(f"data/wealth_final_{gamma}_{chi}.csv", wealth_storage)
    
