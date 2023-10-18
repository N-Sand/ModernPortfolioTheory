import numpy as np
import random, copy
import datetime as dt
from pandas_datareader import data as pdr
import pandas as pd
import yfinance as yfin
import scipy.optimize as optimize
import matplotlib.pyplot as plt
yfin.pdr_override()


def get_data(stocks, start_date, end_date, decay_tau = 252):
  
  stock_data = pdr.get_data_yahoo(stocks, start = start_date, end = end_date)['Close']

  # percent changes
  returns = stock_data.pct_change()

  discount_factor = np.exp(- np.arange(len(returns)-1, -1, -1) / decay_tau)

  # statistics
  #mean_returns = np.sum(returns.multiply(discount_factor, axis=0)) / np.sum(discount_factor)
  mean_returns = returns.mean()
  cov_mat      = returns.cov()

  return mean_returns, cov_mat


def portfolio_performance(weights, mean_returns, cov_mat):
    ret = np.sum(mean_returns * weights) * 252 # the number of trading days in a year
    std = np.sqrt((weights.T.dot(cov_mat)).dot(weights)) * np.sqrt(252)
    return ret, std


def MC_portfolios(N_samples, mean_returns, cov_mat, efficient_weights, epsilon = 0.2):
    N_stocks = mean_returns.size
    p_ret_var = np.zeros((2, N_samples))
    
    for n in range(N_samples):

        weights = copy.copy(random.choice(efficient_weights))
        #weights = copy.copy(efficient_weights[n])
        
        weights += epsilon * np.random.randn(N_stocks)
        weights = np.abs(weights)
        weights /= np.sum(weights)


        ret, std = portfolio_performance(weights, mean_returns, cov_mat)

        p_ret_var[0, n] += ret
        p_ret_var[1, n] += std
    

    return p_ret_var



# calculating the so-called Sharp ratio, which is to be maximized
def negative_SR(weights, mean_returns, cov_mat, riskfreerate = 0.):
    # riskfreerate is the risk-free interest rate. Our portflio must
    # perform better than this to be worth having
    pret, pstd = portfolio_performance(weights, mean_returns, cov_mat)

    return - (pret - riskfreerate) / pstd



def maximize_SR(mean_returns, cov_mat, riskfreerate = 0.0, weight_range = (0., 1.)):
    'minimize -SR in the weights of portfolio'
    N_assets = len(mean_returns)

    # constraints and weight bounds
    args = (mean_returns, cov_mat, riskfreerate)
    constraints = {
        'type' : 'eq',                    # minimizes equation
        'fun'  : lambda x: np.sum(x) - 1  # constrains weights to be normalized
    }
    bounds = tuple(weight_range for i in range(N_assets))

    # minimize
    result = optimize.minimize(negative_SR, 
                      x0 = np.array(N_assets*[1/N_assets]), 
                      args = args,
                      method = 'SLSQP',
                      bounds = bounds,
                      constraints = constraints)
    

    return result


# minimum portfolio variance
def portfolio_variance(weights, mean_returns, cov_mat):
    ret = np.sum(mean_returns * weights) * 252 # the number of trading days in a year
    std = np.sqrt((weights.T.dot(cov_mat)).dot(weights)) * np.sqrt(252)
    return std


def minimize_variance(mean_returns, cov_mat, weight_range = (0., 1.)):

    'minimize portfolio variance in the weights of portfolio'
    N_assets = len(mean_returns)

    # constraints and weight bounds
    args = (mean_returns, cov_mat)
    constraints = {
        'type' : 'eq',                    # minimizes equation
        'fun'  : lambda x: np.sum(x) - 1  # constrains weights to be normalized
    }
    bounds = tuple(weight_range for i in range(N_assets))

    # minimize
    result = optimize.minimize(portfolio_variance, 
                      x0 = np.array(N_assets*[1/N_assets]), 
                      args = args,
                      method = 'SLSQP',
                      bounds = bounds,
                      constraints = constraints)
    

    return result


def get_all_results(mean_returns, cov_mat, riskfreerate = 0., weight_range = (0.,1.), N_steps = 20):

    '''
     Using our simple model, we obtain 2 portfolios and the efficient frontier:
        - the maximum sharp ratio weight  -- the solution
            on the efficient frontier which is tangent to risk=reward line. 
        - the minimum variance weight
            the safest portfolio
        - the efficient frontier
            constructed using these two portfolio weights
    '''


    # maximum sharp ratio
    result_SR = maximize_SR(mean_returns, cov_mat, riskfreerate=riskfreerate, weight_range=weight_range)
    SR_weights = result_SR.x
    SR_max_ret, SR_max_var = portfolio_performance(SR_weights, mean_returns, cov_mat)

    maxSR_dataframe = pd.DataFrame(SR_weights, index = mean_returns.index, columns = ['proportion'])

    # minimum variance 
    result_var = minimize_variance(mean_returns, cov_mat, weight_range=weight_range)
    var_weights = result_var.x
    var_min_ret, var_min_var = portfolio_performance(var_weights, mean_returns, cov_mat)

    minvar_dataframe = pd.DataFrame(var_weights, index = mean_returns.index, columns = ['proportion'])

    # efficient frontier
    efficient_weights = []
    efficient_front = np.zeros((2, N_steps))
    target_returns = np.linspace(var_min_ret, SR_max_ret, N_steps)
    #target_returns = np.linspace(-0.2, 1, N_steps)

    for n, target in enumerate(target_returns):
        result_ef = efficient_frontier(mean_returns, cov_mat, target, weight_range=weight_range)
        efficient_weights.append(np.array(result_ef.x))

        ##### get coords for eff frontier
        ef_ret, ef_var = portfolio_performance(result_ef.x, mean_returns, cov_mat)
        
        efficient_front[0, n] += ef_ret
        efficient_front[1, n] += ef_var

    return maxSR_dataframe, minvar_dataframe, efficient_weights, efficient_front

def portfolio_return(weights, mean_returns):
    ret = np.sum(mean_returns * weights) * 252 # the number of trading days in a year
    return ret
    
def efficient_frontier(mean_returns, cov_mat, return_target, weight_range = (0.,1.)):

    # we choose a return target. We then have a linear combination of maxSR and minvar portfolios
    # which we wish to be greater than our target with minimum variance
    N_assets = len(mean_returns)
    args = (mean_returns, cov_mat)
    constraints = ({
        'type' : 'eq',
        'fun' : lambda x: portfolio_return(x, mean_returns) - return_target # we constrain returns greater than threshold
    },
    {
        'type' : 'eq',
        'fun'  : lambda x: np.sum(x) - 1
    })
    bounds = (weight_range for i in range(N_assets))
    ef_result = optimize.minimize(portfolio_variance, 
                                  np.array(N_assets * [1/N_assets]), 
                                  args = args, 
                                  constraints = constraints, 
                                  bounds = bounds)
    
    return ef_result