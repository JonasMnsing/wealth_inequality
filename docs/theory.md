# Asset Exchange Model (AEM)
- AEMs are a collection of $N$ economic agents, each of which possesses some amount of wealth $w$
- Agents engage in pairwise transactions according to idealized rules
- Interactions usually converge the total number of agents $N$ and their total wealth $W$, i.e. for a closed economy
- *Yard Sale Model (YSM):* In an interaction, the losing agent is selected with even odds, but the amount lost is a fraction $\beta$ of the wealth of the poorer agent
- Idea: People engage in transaction for which the amount at stake is strictly less than their own total wealth
- This will always end up in oligarchy (one agent gets all the wealth). Richer agents are able to withstand a longer string of losses 
- One might also add an uneven (biased) seletion of the losing agents
- A redistribution term might represent taxes preventing oligarchy

## Basics
- In the simplest version of YSM, each agent is just defined by its wealth $w$, which is positive
- At micro level, the economy evolves in time by sequential pairwise transactions
- At macro level, in continuum limit, the wealth distribution is described by the agent density function $P(w,t)$. This function is defined so that the following integral equals the number of agents with wealth between $a$ and $b$ 
$$\int_{a}^{b} dw P(w,t)$$
- The first two moments of the agent density function relate to:
  - First Moment (Number of Agents)
  $$N = \int_0^{\infty} dw P(w,t)$$
  - Second Moment (Total Wealth)
  $$W = \int_0^{\infty} dw P(w,t) \cdot w$$
- At micro level:
  - A pair of agents with wealth $w_i$ and $w_j$ is selected at random
  - The direction of transaction is given by a fair coin flip
  - In this transaction $\Delta w = \sqrt{\gamma \cdot \Delta t} \cdot min(w_1,w_2) \eta$ is transfered.
  - Here $\beta = \sqrt{\gamma \cdot \Delta t}$ is based on the characteristic time associated with a transaction and a parameter $\gamma$. The coin flip is modeled by the stoachstic variable $\eta \in \{-1,+1\}$
  - First agent's final wealth: $w_1 = w_1 + \Delta w$