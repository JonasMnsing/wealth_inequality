import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pandas as pd
from scipy.interpolate import interp1d
from scipy import integrate
import random

### GDP ###

def calc_gdp(GDP, d_GDP, consumption_annual, consumption, investment_annual, investment):
    
    """
    In closed economy: GDP = C + I + G
    C : Consumption/period
    I : investment/period
    G : Government expenditure / G = 0
    """
    
    # Append this years consumption to list of annual consumption: 
    consumption_annual.append(consumption)
    # Append this years investments to list of annual investments: 
    investment_annual.append(investment)
    


    # Calculate this years GDP:
    Y = consumption + investment

    # Append Y to list of past GDP values: 
    GDP.append(Y)

    # Calculate change in GDP between this year and last year
    
    if len(GDP) >= 2:
        d_GDP = GDP[-1]/GDP[-2]

    return consumption_annual, investment_annual, GDP, d_GDP

### Consumption/Transaction Part ###

def transaction(wealth, zeta, chi, kappa):
    
    """
    Transaction between two agents. Both are selected randomly from the wealth distribution.
    
    Parameter:
    wealth = Wealth distribution, array of shape (1,N)
    zeta = Wealth-Attained Advantage
    chi = redistribution rate
    kappa = downward shift - allows negative wealth 

    """
    
    # Number of Agents:
    N = len(wealth)
    # Mean Wealt
    w_mean = np.mean(wealth)

    # Select agents, but not the the same two times:
    i = 1 
    j = 1
    while i == j:
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)

    # If negative wealth possible (kappa != 0), add loan S
    S = kappa*w_mean
    
    # Wealth of both agents:
    w_i = wealth[i] + S
    w_j = wealth[j] + S

    max_w = max([w_i, w_j])
    
    # Wealth transfered
    cash = min(w_i, w_j)*0.1

    # Coin Bias:
    bias = zeta*(w_i - w_j)/max_w

    p = (bias + 1)/2

    eta = np.random.binomial(1, p)

    if eta == 1:
        wealth[i] = wealth[i] + cash - S
        wealth[j] = wealth[j] - cash - S

    else:
        wealth[i] = wealth[i] - cash - S
        wealth[j] = wealth[j] + cash - S

    return wealth, cash

def tax(wealth, chi):
    
    avrg = np.mean(wealth)
    wealth = wealth + chi*(avrg - wealth)
    
    return wealth

### Investment Part ###

def shareholder_perc(wealth, income_dependency):
    
    """
    Function calculates an array of N values
    The array corresponds to the probabilities at which each agent want to invest 
    """

    if income_dependency == True:
        
        # Investment probability depending on difference to median wealth:
        bias = (wealth - np.median(wealth))/max(wealth)
        # Biased coin flip
        p = (bias + 1)/2

        # Array of probabilitys with Values 0 or 1:
        perc_array = np.random.binomial(1, p)

    else:
        # Each agent want to invest
        perc_array = np.ones(len(wealth))
        

    return(perc_array)  

def invest(wealth, investments, shareholders, V, PI):
    
    """
    wealth: array
    investments: array
    mask: array
    V: global investment index, 0 <= V <= 1
    PI: part of w, agents invest 0 <= PI <= 1
    """

    # Calculate mask or the desire of the agents to invest
    mask = V*shareholders >= np.random.rand(N)

    # Investments
    inv = PI*wealth
    
    # Agents that dont fit the mask wont invest:
    inv[np.where(np.invert(mask))] = 0
    
    # Transfer wealth between the accounts:
    investments = investments + inv  
    
    wealth = wealth - inv

    inv_total = sum(inv)

    return(wealth, investments, inv_total)

"""
def investment(wealth, investments, V, Pi):
    # The investment dynamic is based on a Gaussian multiplicative process
    
    # V: global investment index of the entire society
    # PI: percentage of wealth the agents invest - array
    
    # Individual investment index (standard deviation):
    r = V*Pi

    investments = investments + r*np.random.randn(len(wealth))*wealth
    wealth = wealth - r*np.random.randn(len(wealth))*wealth
    
    return(wealth, investments)
"""

def dividend(wealth, investments, rate):
    
    # Individual dividend:
    d = rate*investments
    wealth = wealth + d
    investments = investments - d

    return(wealth, investments)

def economy_dividend(wealth, investments, rate):

    # Economy dividend:
    x = investments*rate

    investments = investments - x
    wealth = wealth + sum(x)/len(x)

    return(wealth, investments)

def returns(investments, GDP_growth):
    
    investments = investments*GDP_growth
    
    return(investments)


### Cumulative and Gini Part ###

def cumulative(wealth, N):

    # Cumulative Wealth + Normalization:
    wealth_cum_norm=np.zeros(N+1)
    wealth=np.sort(wealth)
    wealth_cum = np.cumsum(wealth)
    wealth_cum_norm[1:] = wealth_cum/wealth_cum[-1]

    # Cumulative Population + Normalization:
    N_cum = np.linspace(0,1,N+1)
    #N_cum = np.cumsum(range(N))
    N_cum_norm = N_cum/N_cum[-1]
    # Difference and interpolation:
    # h = abs(wealth_cum_norm - N_cum_norm)
    f_1 = interp1d(N_cum_norm, N_cum_norm)#,fill_value="extrapolate")
    f_2 = interp1d(N_cum_norm, wealth_cum_norm)#,fill_value="extrapolate")

    return(N_cum_norm, wealth_cum_norm, f_1, f_2)
        
def gini(f_1, f_2, dec=4):
    
    # Gini Coeff
    G = round(2*(integrate.quad(f_1, 0, 1)[0] - (integrate.quad(f_2, 0, 1)[0])), dec)    
    
    return(G)


### Save Data ###

def save_cum(cum_wealth, cum_population, t, zeta, chi, kappa, G, v, PI, div, rep):
        
    # Save cumulative Values as csv:
    cum_df = pd.DataFrame(cum_wealth,cum_population)
    cum_df.to_csv('data/cumulative_data/N{}_t{}_z{}_c{}_k{}_v{}_PI{}_div{}_rep{}.csv'.format(
                N, t, zeta, chi, kappa, v, PI, div, rep))

    # Save parameter as csv:
    par_s = pd.Series(data=[N, t, zeta, chi, kappa, G], index=['Agents', 'Transactions', 'Zeta', 'Chi', 'Kappa', 'Gini'])
    par_s.to_csv('data/parameter/N{}_t{}_z{}_c{}_k{}_v{}_PI{}_div{}_rep{}.csv'.format(
                N, t, zeta, chi, kappa, v, PI, div, rep))

def save_array(wealth, investments, gdp, annual_inv, annual_cons, N, t, zeta, chi, kappa, v, PI, div, rep):
    
    # Save wealth as csv:
    w_df = pd.DataFrame(wealth)
    w_df.to_csv('data/arrays/w_N{}_t{}_z{}_c{}_k{}_v{}_PI{}_div{}_rep{}.csv'.format(
                N, t, zeta, chi, kappa, v, PI, div, rep))

    # Save investments as csv:
    i_df = pd.DataFrame(investments)
    i_df.to_csv('data/arrays/i_N{}_t{}_z{}_c{}_k{}_v{}_PI{}_div{}_rep{}.csv'.format(
                N, t, zeta, chi, kappa, v, PI, div, rep))

    # Save GDP/annual_consumption/annual_investments as csv
    d = {'GDP': gdp, 'Annual_Consumption': annual_cons, 'Annual_Investments': annual_inv}
    gdp_df = pd.DataFrame(data=d)
    gdp_df.to_csv('data/arrays/gdp_N{}_t{}_z{}_c{}_k{}_v{}_PI{}_div{}_rep{}.csv'.format(
                N, t, zeta, chi, kappa, v, PI, div, rep))

def save_parameter(wealth, statistics, p90, t, zeta, chi, kappa, G, v, PI, div, rep):
    
    # Save parameter as csv:
    par_s = pd.Series(data=[N, t, zeta, chi, kappa, G, p90], index=['Agents', 'Transactions', 'Zeta', 'Chi', 'Kappa', 'Gini', '90%'])
    par_s = par_s.append(statistics)
    par_s.to_csv('data/parameter/N{}_t{}_z{}_c{}_k{}_v{}_PI{}_div{}_rep{}.csv'.format(
                N, t, zeta, chi, kappa, v, PI, div, rep))


# Number of agents
N = 50

# Starting Wealth
w = np.ones(N)
w_array = w

# Number of transactions
t = 50000

# Coefficients
zeta = [0.1] # [0.9] # [0.1]
chi = [0.09] # [0.01] # [0.09]
kappa = [0]

# Initial Consumption:
consumption = 0
consumption_annual = []

# Initial Investment:
investment_annual = []
inv = np.zeros(len(w))
inv_array = inv

# Initial GDP:
GDP = []
d_GDP = 1

# Investment Parameter:
V = [round(x*0.03,2) for x in range(0,34)]
PI_list = [0.1] # [round(x*0.03,2) for x in range(0,34)] # 0.1 # Percentage invested
div_list = [0.05] # [round(x*0.03,2) for x in range(0,34)]
# e_div = 0.05

# Empty lists for stat values:
q_50 = []
q_90 = []
avg = []
std = []

W_total = []
I_total = []

Gini_list = []

rep = 10

for div in div_list:
    for v in V:
        for PI in PI_list:
            for k in kappa:
                for c in chi:
                    for z in zeta:
                        for j in range(rep):
                            for i in range(t):
                                
                                # Do a transaction and save amount of wealth in dw
                                w, dw = transaction(w, z, c, k)
                                # Add up consumptions
                                consumption = consumption + abs(dw)

                                # After "one year" defined by N transactions
                                if i % N == 0:
                                    
                                    # Agents pay tax:
                                    w = tax(w, c)

                                    # Investment Returns: 
                                    inv = returns(inv, d_GDP)
                                    
                                    if i % 10*N == 0:

                                        w, inv = dividend(w, inv, div)

                                        # w, inv = economy_dividend(w, inv, e_div)

                                        shareholder = shareholder_perc(w, income_dependency=False)

                                        # Agents invest:
                                        w, inv, d_inv = invest(w, inv, shareholder, v, PI)
                                    
                                    # Calculate annual consumption, investments and GDP / GDP_change
                                    consumption_annual, investment_annual, GDP, d_GDP = calc_gdp(GDP, d_GDP, consumption_annual, consumption, investment_annual, d_inv)
                                    
                                    # Reset Consumption
                                    consumption = 0

                                    # Add new periods wealth and investment arrays to old arrays:
                                    w_array = np.vstack((w_array, w))
                                    inv_array = np.vstack((inv_array, inv))

                            # Final Statistics:
                            w_stats = pd.Series(w + inv).describe()

                            # 90% percentile
                            p90 = np.percentile((w + inv), 90)

                            # Calc Cumulative Data and interpolation:    
                            N_cum, w_cum, f_1, f_2 = cumulative(w, N)
                            # Calc Gini Coeff:
                            G = gini(f_1, f_2)

                            # Save Cumulative Data:
                            save_cum(w_cum, N_cum, t, z, c, k, G, v, PI, div, j)

                            # Save investment, wealth, GDP, consumption
                            save_array(w_array, inv_array, GDP, investment_annual, consumption_annual, N, t, z, c, k, v, PI, div, j)
                            
                            # Save other parameter
                            save_parameter(w, w_stats, p90, t, z, c, k, G, v, PI, div, j)

                            # New Initial Values:
                            # Initial Consumption:
                            consumption = 0
                            consumption_annual = []

                            # Initial Investment:
                            investment_annual = []
                            inv = np.zeros(len(w))
                            inv_array = inv

                            # Initial GDP:
                            GDP = []
                            d_GDP = 1

                            # Initial Wealth:
                            w = np.ones(N)
                            w_array = w
