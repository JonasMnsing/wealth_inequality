# Modelling Wealth Distribution
## An agent-based Model

- Population contains $N$ agents each with some positive amount of wealth $w$
- A specific agent 1 with wealth $w_1$ transacting with a randomly selected agent 2 with wealth $w_2$
- In this transaction, the wealth of agent 1 is increased by $\Delta w$, while that of agent 2 is decreased by $\Delta w$
- $\Delta w$ may be positive or negative and is described by the statistical process
  $$\Delta w = \sqrt{\gamma \Delta t} \cdot min(w_1,w_2) \eta + \chi (\frac{W}{N} - w)\Delta t$$
- Coin flip is modeled by $\eta$. This value is positive if wealth is moving from agent 2 to agent 1, and negative if it is moving from agent 1 to agent 2. The expected value of $\eta$ is
  $$E[\eta] = \zeta \sqrt{\frac{\Delta t}{\gamma}} \cdot (\frac{w_1-w_2}{W/N})$$
