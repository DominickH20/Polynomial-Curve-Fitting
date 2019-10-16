import numpy as np
import matplotlib.pyplot as plt
from Data_Generator import generate_dataset
from Data_Generator import generate_sin_dataset

#Fit coefficients using gradient descent
def fit_polynomial(x_train, y_train, order = 1, epochs = 5, lr = .1, reg = 0):
    coefs = np.zeros(order+1)

    for n in range(epochs):
        #Compute predictions and error
        y_predict = np.asarray([predict(coefs, x) for x in x_train])
        print("MSE: ", MSE(coefs, x_train, y_train))

        #Construct Gradient
        gradient = np.zeros(len(coefs))
        for k in range(len(coefs)):
            grad = np.sum(-2*(y_train - y_predict)*(x_train**k))/len(x_train)
            reg_opt = reg*2*coefs[k]
            gradient[k] = grad + reg_opt

        #Gradient clipping to prevent exploding gradient
        clip_gradient(gradient,1)

        #Update coefficients
        coefs = coefs - learning_rate(0, lr, epoch)*gradient

    return coefs

#Fit coefficients using deterministic matrix solution
def deterministic_fit(x_train, y_train, order = 1):
    #Compute Constant Matrix
    X = np.zeros((len(x_train),order+1))
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j] = x_train[i]**j

    #Compute helper matrices to solve for coefficients
    XT = np.transpose(X)
    Y = np.transpose(np.asarray([y_train]))
    XXinv = np.linalg.inv(np.matmul(XT,X))
    B = np.dot(np.matmul(XXinv,XT),Y)

    return np.transpose(B)[0]

#Compute MSE of predictions
def MSE(coefs, X_data, Y_data):
    y_predict = np.asarray([predict(coefs, x) for x in X_data])
    error = np.sum((Y_data - y_predict)**2)/len(X_data)
    return error

#Output prediction for a given input
def predict(coefs, x):
    points = np.asarray([x**i for i in range(len(coefs))])
    return np.sum(points*coefs)

#Decaying learning rate
def decay_learning_rate(base,epoch):
    if epoch == 0:
        return base
    else:
        return base * (100/epoch)

#learning rate wrapper for cleanliness
def learning_rate(type_key, lr, epoch):
    if type_key == 0:
        return lr
    elif type_key == 1:
        return decay_learning_rate(lr, epoch)

#Normalze gradient to a specified threshold
def clip_gradient(gradient, threshold):
    norm = np.linalg.norm(gradient)
    if norm > threshold:
        gradient = gradient * (threshold/norm)


# main
if __name__ == '__main__':
    frame_size = 2

    #generate artificial data set
    true_coefs = np.asarray([-5,0,7,3])
    data_set, components = generate_dataset(true_coefs,100,-frame_size,
                                            frame_size, noise = 0)
    x_train, y_train = data_set[:,0],data_set[:,1]
    #x_train, y_train = generate_sin_dataset(frame_size)


    #Fit the Model
    fit_coefs = fit_polynomial(x_train, y_train ,order = 3,
                                epochs = 1000, lr = 0.01, reg = 0)
    # fit_coefs = deterministic_fit(x_train,y_train,order=9)

    #Print final coefficients and error
    print("FITTED: ", fit_coefs)
    print("MSE: ", MSE(fit_coefs,x_train,y_train))

    #Plot data and Model function
    plt.scatter(x_train, y_train, color = 'blue')
    plt.plot(np.arange(-frame_size,frame_size,.01),
            [predict(fit_coefs, x) for x in np.arange(-frame_size,frame_size,.01)],
            color = 'red')
    plt.show()
