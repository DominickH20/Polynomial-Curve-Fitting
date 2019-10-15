import numpy as np
import matplotlib.pyplot as plt

# randomly generate polynomial coefficients from a uniform distribution
# specified by parameters low,high
def generate_coefficents(k, low, high):
    return np.asarray([np.random.uniform(low = low, high = high) for i in range(k)])

# Given a set of polynomial coefficients, generate a data set over the given x
#range of specified size
def generate_dataset(coefficients, sample_size, x_low, x_high, noise = .05):
    data_set = np.zeros((sample_size,2))
    components = np.zeros((sample_size,len(coefficients)))
    for i in range(sample_size):
        x = np.random.uniform(low = x_low, high = x_high)
        points = np.asarray([x**i for i in range(len(coefficients))])
        components[i] = points*coefficients
        data_set[i] = np.asarray([x,np.sum(components[i]) + noise * np.random.normal()])
    return data_set, components

# main
if __name__ == '__main__':
    # Get Data
    coefs = generate_coefficents(2,-1,1)
    data, components = generate_dataset(coefs, 20, -1, 1)

    #Plot Data
    plt.scatter(data[:,0],data[:,1], color = 'blue')
    plt.show()
