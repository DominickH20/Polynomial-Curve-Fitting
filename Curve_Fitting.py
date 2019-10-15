import numpy as np
import matplotlib.pyplot as plt
from Data_Generator import generate_dataset

def fit_polynomial(x_train, y_train, order = 1, epochs = 5, lr = .1, reg = 0):
    coefs = np.zeros(order+1)

    for n in range(epochs):
        y_predict = np.asarray([predict(coefs, x) for x in x_train])
        print("MSE: ", MSE(coefs, x_train, y_train))

        gradient = np.zeros(len(coefs))
        for k in range(len(coefs)):
            grad = np.sum(-2*(y_train - y_predict)*(x_train**k))/len(x_train)
            reg_opt = reg*2*coefs[k]
            gradient[k] = grad + reg_opt

        #Gradient clipping
        clip_gradient(gradient,1)

        #coefs = coefs - decay_learning_rate(lr,n)*gradient
        coefs = coefs - lr*gradient
        #print("COEFS: ",coefs)

    return coefs

def deterministic_fit(x_train, y_train, order = 1):
    X = np.zeros((len(x_train),order+1))
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j] = x_train[i]**j


    XT = np.transpose(X)
    Y = np.transpose(np.asarray([y_train]))
    XXinv = np.linalg.inv(np.matmul(XT,X))
    B = np.dot(np.matmul(XXinv,XT),Y)
    return np.transpose(B)[0]


def MSE(coefs, X_data, Y_data):
    y_predict = np.asarray([predict(coefs, x) for x in X_data])
    error = np.sum((Y_data - y_predict)**2)/len(X_data)
    return error

#WRONG
def clip_gradient(gradient, threshold):
    #print("B:",np.linalg.norm(gradient))
    norm = np.linalg.norm(gradient)
    if norm > threshold:
        gradient = gradient * (threshold/norm)
    #print("A:",np.linalg.norm(gradient))

def predict(coefs, x):
    points = np.asarray([x**i for i in range(len(coefs))])
    return np.sum(points*coefs)

def decay_learning_rate(base,epoch):
    if epoch == 0:
        return base
    else:
        return base * (100/epoch)

def sin_data(frame):
    x = np.arange(-frame,frame,.5)
    y = np.sin(x)
    return x,y

# main
if __name__ == '__main__':
    frame_size = 2

    #generate artificial data set
    true_coefs = np.asarray([-5,0,7,3])
    data_set, components = generate_dataset(true_coefs,100,-frame_size,
                                            frame_size, noise = 0)
    x_train, y_train = data_set[:,0],data_set[:,1]
    #x_train, y_train = sin_data(frame_size)

    fit_coefs = fit_polynomial(x_train, y_train ,order = 3,
                                epochs = 1000, lr = 0.01, reg = 0)
    print("FITTED: ", fit_coefs)
    print("MSE: ", MSE(fit_coefs,x_train,y_train))

    # fit_coefs = deterministic_fit(x_train,y_train,order=9)
    # print("FITTED: ", fit_coefs)
    # print("MSE: ", MSE(fit_coefs,x_train,y_train))

    plt.scatter(x_train, y_train, color = 'blue')
    plt.plot(np.arange(-frame_size,frame_size,.01),
            [predict(fit_coefs, x) for x in np.arange(-frame_size,frame_size,.01)],
            color = 'red')
    plt.show()
