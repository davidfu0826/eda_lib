from functools import reduce
import numpy as np

# https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
def _change_input_to_dict_style(func):
    def wrapper(dict_input, dict_input2):
        x_mean, y_mean = dict_input["mean"], dict_input2["mean"]
        x_var,  y_var  = dict_input["var"],  dict_input2["var"]
        n,      m      = dict_input["n"],    dict_input2["n"]
        new_mean, new_var, new_n = func(x_mean, y_mean, x_var, y_var, n, m)
        return {
            "n":    new_n,
            "mean": new_mean,
            "var":  new_var,
            "std":  new_var**0.5
        }     
    return wrapper
    
#@_change_input_to_dict_style
def combine_moments_of_two_distributions(x_mean, y_mean, x_var, y_var, n, m):
    """
    Returns the mean, and variance of two distributions when given the mean, and variance separately.
    
        Parameters:
            x_mean (float): Mean of x
            y_mean (float): Mean of y
            x_var (float): Variance of x
            y_var (float): Variance of y
            n (int): Population size of x
            m (int): Population size of y
    """
    n, m = np.array(n, dtype=np.longdouble), np.array(m, dtype=np.longdouble)
    new_size = n + m
    
    new_mean = (n*x_mean + m*y_mean) / new_size

    term_1 = ((n-1)*x_var+(m-1)*y_var) / (new_size-1)
    term_2 = (n*m*(x_mean - y_mean)**2) / (new_size*(new_size-1))
    new_var = term_1 + term_2
    
    return new_mean, new_var, int(new_size)

def combine_moments_of_distributions(stats):
    """
    Returns the mean, and variance of n distributions when given a list of the mean, and variance separately.
    
        Parameters:
            stats (list): List of dictionaries, each dictionary with the keys: 
                n (int): Number of sampels in distribution          
                mean (float): Mean of distribution size
                var (float): Variance of distribution
                std (float): Standard deviation of distribution
    """
    return reduce(_change_input_to_dict_style(combine_moments_of_two_distributions), stats)

#                                           Combined size          Combined mean    Combined variance        Combined std
# Number of distributions: 186270 - Error:              0  2.359223927328458e-16  0.10218996188646703   0.117419596825389
# Number of distributions:  37254 - Error:              0  1.804112415015879e-16  0.02143040802382890   0.022195687784406
# Number of distributions:   3726 - Error:              0  4.163336342344337e-17  0.00217263779746044   0.002204576931424
# Number of distributions:    373 - Error:              0  4.163336342344337e-17  0.00021823748596073   0.000221000897021
# Number of distributions:     38 - Error:              0  1.387778780781446e-17  2.2758779406495e-05   2.30423410625e-05
# Number of distributions:      4 - Error:              0  1.387778780781446e-17  1.9515551685501e-06   1.97582831146e-06

# Ground truth:                                    372539     0.0742191752859584  0.24389423150538608   0.493856488775217