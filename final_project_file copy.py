'''The code below is a very simple implementation of solving a differential equations with the help of neural network
and this segment of machine learning is also known as Physics Informed Neural Networks(PINNs). Because whatever physical 
property we want our system,we model a differential equation for that property and then implement that as a loss function
for our network. 
There are many other methods to incorporate physics and symmetry related to physics such as by augmenting the data or 
choosing a architecture that forces that physics by construction, one of the method is to craft a lost function using
our differential equation and let our plain neural network learn.
'''



'''The code below is divide into six parts:
                            1: Setting up and importing the needed libraries for mathematics and plotting the results
                            2: Defining some  functions that we will be needing in the calculations ahead
                            3: Defining a neural network architecture that does the forward pass of the input data,
                               it is just a series of basic matrix multiplication.
                            4: This section consists of two parts 
                            Part 4:A:
                                    This section defines the trial function,the cost function and the get gradients function
                                    the analytic solution function for the 1st order ODE
                            Part 4:B:
                                    Thjis section defines the trial solution,get_gradients function and the cost function,
                                    the analytic solution for the 2nd order ODE.
                            5. This section contains the function which takes the input such as number of iterations,
                            paramneters input vectors etc, and tries to learn the solution
                            This section also asks which degree of ode we want to solve,takes the coefficients and 
                            intial condition of our system.
                            6. This section calls the program based on our input and also plots the result

'''

'''Mostly people train this models using deeplearning libraries such as tensorflow, pytorch etc. which makes this neural networks kind of black box.
Also they use automatic differentiation to calculate the gradients of the loss function with respect to the parameters and inputs.

Whereas in the following code we haven't used any deep learning library such as pytorch ,autograd or jax to implement the calculation.
Although the method of not using library makes our code slow as compared to the optimized matrix calculus but it lets us see how the networks learn
'''


''' Part 1'''

import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt

'''Part 2'''
#-- We have chosen sigmoid(z) as our regukarization function
def sigmoid(z):   # this function takes an input and return sigmoid(x)
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):   #this function returns the derivative of sigmoid function
    s = sigmoid(z)
    return s * (1 - s)

def sigmoid_double_prime(z):  #this function returns the second order derivative of sigmoid function
    s = sigmoid(z)
    s_prime = s * (1 - s)
    return s_prime * (1 - 2*s)

def sigmoid_triple_prime(z):  #this function returns the third order derivative of sigmoid function
    s = sigmoid(z)
    s_prime = sigmoid_prime(z)
    s_double_prime = sigmoid_double_prime(z)
    return s_double_prime * (1 - 2*s) - 2 * (s_prime**2)

# --- COMMON NEURAL NETWORK ---
''' Part 3'''
'''This section of code defines a neural network architecture which takes the input as parameter and the input vector(x) for which we have to predict the value y.
The parameter input contains two list i.e P[0]: it contains the weights and biases of the hidden layer, the P[0] has two columns and 10 rows
                                                                                                                             column0: contains the biases of the hidden layer
                                                                                                                             column1: contains the weights of the hidden layer
                                          P[1]: it contains the weights and bias of the output layer,P[1] has 1 row and 11 columns
                                                                                                    column0: contains the bias of the output layer
                                                                                                    column[1 to 1o]: contains the weights of the output layer

'''
'''Our Neural network architecture has three layers 1: the input layer
                                                    2: the hidden layer
                                                    3: the output layer
Our neural network takes the input as input vector x and predicts a vector of values let's say N(x) which in combination with other function
make our expected solution of the differential equation.
'''
def neural_network(params, x):
    """A 1-hidden-layer neural network"""
    w_hidden = params[0] # w_hidden contains the parameters for the hidden layers(includes bias and weights)
    w_output = params[1] #w_output contains the parameters for the output layers(includes bias and weights)

    num_values = np.size(x) #it the the number of element in the input vector x #lt's assume that num_values=10
    x = x.reshape(1, -1)  #it reshapes the vector into a matrix with row=1 column=num_values(10)

    x_input = np.concatenate((np.ones((1,num_values)), x ), axis = 0) #adding a row of 1s to our row of x's
    z_hidden = np.matmul(w_hidden, x_input) #Does the matrix multiplication between the hidden parameters and the x_input and gives us matrix of dimension 10x10
    #z_hidden is also known as the affine function in literatures
    x_hidden= sigmoid(z_hidden) # this takes the z_hidden and applies sigmoid function to it and gives the activation of the hidden layer

    x_hidden_with_bias = np.concatenate((np.ones((1,num_values)), x_hidden ), axis = 0) #adding a row of 1s to our row of x's
    z_output = np.matmul(w_output, x_hidden_with_bias)#Does the matrix multiplication between the output parameters and the x_input and gives us matrix of dimension 1x10
    x_output = z_output 

    return z_hidden, x_output # returns the affine function and the predicted output of our neural Network N(x).
'''Part 4:A'''
'''this section of code defines the function required to formulate the given ode in terms of the neural network output,
also it defines the cost function (also know ans loss function)
and calculates the gradien of the cost function with different network parameters using chain rule
'''
# --- 1ST ORDER ODE FUNCTIONS (g' + gamma*g = 0) ---
def g_trial_1st_order(x, params, g0=10):
    _z_hidden, z_output = neural_network(params,x)
    return g0 + x * z_output  # returns the g_trial function which we will be trying to fit through our differential equation

def f_1st_order(x, g_trial, gamma=2): #returns the right side of the ode
    return -gamma * g_trial

def cost_function_1st_order(P, x, g0=10, gamma=2):#this function takes input as parameters(P),x intial condition(g_0),gamma and returns the loss for that particular set of parameters.
    z_hidden, z_output = neural_network(P,x)
    g_t = g0 + x * z_output #return the g_trial in terms of output of the neural network and input x

    w_h = P[0][:, 1:2] #w_h=weights corresponding to the hidden layer
    v = P[1][:, 1:]#v=weights corresponding to the output layer
    sig_prime = sigmoid_prime(z_hidden)#activation fo the hidden layer
    
    d_z_o_dx = np.dot(v, sig_prime * w_h) #the rate of change of output of neural network wrt to input i.e (dN/dx)
    d_g_t = z_output + x * d_z_o_dx #the rate of change of the trial function with respect to input i.e(dg/dx)

    func = f_1st_order(x, g_t, gamma)# this returns a vector named error which contains the difference between the predicted and expected value
    err_sqr = (d_g_t - func)**2 #this calculates the sum of all the elements in error vector
    return np.mean(err_sqr)# this calculates the mean and gives us the cost or the loss
''' this section of our code calculates the gradient of our loss or cost function wrt to different parameter of the network which will ne used 
in backpropagating the error (backpropagation algorithm)
and hence in turn traing the network'''
def get_gradients_1st_order(P, x, g0=10, gamma=2):
    num_values = np.size(x)
    x = x.reshape(1, num_values) 

    w_hidden, w_output = P[0], P[1]
    b_h, w_h = w_hidden[:, 0:1], w_hidden[:, 1:2]
    b_o, v = w_output[:, 0:1], w_output[:, 1:]

    x_input = np.concatenate((np.ones((1, num_values)), x), axis=0)
    z_h = np.matmul(w_hidden, x_input)
    a_h = sigmoid(z_h)

    a_h_with_bias = np.concatenate((np.ones((1, num_values)), a_h), axis=0)
    z_o = np.matmul(w_output, a_h_with_bias)

    sig_prime = sigmoid_prime(z_h)
    sig_double_prime = sigmoid_double_prime(z_h)

    g_t = g0 + x * z_o
    d_z_o_dx = np.dot(v, sig_prime * w_h)
    d_g_t = z_o + x * d_z_o_dx

    '''till here its the same as the previous part defined stated in the neural_network(section)'''

    err = d_g_t + gamma * g_t #err is a vector that contains the different error values corresponding to different different Xi 's
    '''the gradients are calculated using chain rule of derivatives and every derivative has common term that is (dc/derr)
    So we calculate that separately'''
    common_term = (2.0 / num_values) * err #common term=dc/derr

    d_err_d_b_o = 1 + gamma * x #this calculates the rate of change of error wrt output bias b_o 
    grad_b_o = np.sum(common_term * d_err_d_b_o) #this calculates the gradient of the cost function wrt to output bias b_o

    d_err_d_v = a_h * (1 + gamma * x) + x * sig_prime * w_h#this calculates the rate of change of error wrt output weights v 
    grad_v = np.sum(common_term * d_err_d_v, axis=1)#this calculates the gradient of the cost function wrt to output weights v 

    d_err_d_b_h = v.T * (sig_prime * (1 + gamma * x) + x * sig_double_prime * w_h)#this calculates the rate of change of error wrt hidden bias(b_h) 
    grad_b_h = np.sum(common_term * d_err_d_b_h, axis=1)#this calculates the gradient of the cost function wrt to hidden bias (b_h)


    d_err_d_w_h = v.T * (sig_prime * x * (1 + gamma * x) + sig_double_prime * w_h * (x**2))#this calculates the rate of change of error wrt hidden weights(w_h) 
    grad_w_h = np.sum(common_term * d_err_d_w_h, axis=1)#this calculates the gradient of the cost function wrt to hidden weights (w_h)
    
    grad_P0 = np.stack((grad_b_h, grad_w_h), axis=-1)#stacks the gradient of the hidden layer
    grad_P1 = np.concatenate(([grad_b_o], grad_v)).reshape(1, -1)
    
    return [grad_P0, grad_P1] #returns the desired value of (dc/dv,dc/db_o,dc/dw_h,dc/db_h)

def g_analytic_1st_order(x, gamma=2, g0=10):
    return g0*np.exp(-gamma*x)#returns the analytic solution so as to compare the performance of our predicted solution


'''Part 4:B'''

# --- 2ND ORDER ODE FUNCTIONS (g'' + lambda*g' + mu*g = 0) ---

'''this section of code defines the function required to formulate the given ode in terms of the neural network output,
also it defines the cost function (also know ans loss function)
and calculates the gradien of the cost function with different network parameters using chain rule
'''
def g_trial_2nd_order(x, params, g_0=2, g_0_dash=1):#
    _z_hidden, x_output = neural_network(params,x)
    return g_0 + (g_0_dash*x) + ((x**2)*x_output)# returns the g_trial function which we will be trying to fit through our 2nd order differential equation

def cost_function_2nd_order(P, x, lambda1=5, mu1=6, g_0=2, g_0_dash=1):#this function takes input as parameters(P),x intial condition(g_0),gamma and returns the loss for that particular set of parameters.
    w_h=P[0][:,1:2] #w_h=weights corresponding to the hidden layer
    v=P[1][:,1:]#v=weights for the output layer
    
    z_hidden,x_output=neural_network(P,x)
    
    sig_prime=sigmoid_prime(z_hidden)
    sig_double_prime = sigmoid_double_prime(z_hidden)#activation function of the hidden layer
    
    d_x_o_dx=np.dot(v,sig_prime*w_h)#returns the rate of change of neural network output wrt to input vector x
    d_g_dx=g_0_dash+2*x*x_output+(x**2)*(d_x_o_dx)#returns the rate of change of the trial function wrt to input vector x
    
    d2_x_o_dx2=np.dot(v,sig_double_prime*(w_h*w_h))#calculates the second order derivative of the neural network output wrt input vector x
    d2_g_dx2=(2*x_output)+(4*x*d_x_o_dx)+(x**2*d2_x_o_dx2)#calculates rate of change of trial function wrt to input vector x
    
    err = d2_g_dx2 + (lambda1 * d_g_dx) + (mu1 * (g_0 + (g_0_dash * x) + (x**2 * x_output)))# err is a vector that return an array containg the differnce between the predicted values and expected values
    errsq=err**2
    return np.mean(errsq) #return the mean square error of the err vector also known as the loss function


''' this section of our code calculates the gradient of our loss or cost function wrt to different parameter of the network which will ne used 
in backpropagating the error (backpropagation algorithm)
and hence in turn traing the network'''

def get_gradients_2nd_order(P, x, lambda1=5, mu1=6, g_0=2, g_0_dash=1):
    num_values = np.size(x)
    x = x.reshape(1, num_values) 

    w_hidden, w_output = P[0], P[1]
    b_h, w_h = w_hidden[:, 0:1], w_hidden[:, 1:2]
    b_o, v = w_output[:, 0:1], w_output[:, 1:]

    x_input = np.concatenate((np.ones((1, num_values)), x), axis=0)
    z_h = np.matmul(w_hidden, x_input)
    a_h = sigmoid(z_h)

    a_h_with_bias = np.concatenate((np.ones((1, num_values)), a_h), axis=0)
    z_o = np.matmul(w_output, a_h_with_bias)
    x_output = z_o

    sig_prime = sigmoid_prime(z_h)
    sig_double_prime = sigmoid_double_prime(z_h)
    sig_triple_prime = sigmoid_triple_prime(z_h)

    d_x_o_dx = np.dot(v, sig_prime*w_h)
    d_g_dx = g_0_dash + 2*x*x_output + (x**2)*(d_x_o_dx)
    
    d2_x_o_dx2 = np.dot(v, sig_double_prime*(w_h**2))
    d2_g_dx2 = (2*x_output) + (4*x*d_x_o_dx) + (x**2*d2_x_o_dx2)

    '''till here its the same as the previous part defined stated in the neural_network(section) and cost function section'''
    
    err = d2_g_dx2 + (lambda1 * d_g_dx) + (mu1 * (g_0 + (g_0_dash * x) + (x**2 * x_output)))#err is a vector that contains the different error values corresponding to different different Xi 's
    
    '''the gradients are calculated using chain rule of derivatives and every derivative has common term that is (dc/derr)
    So we calculate that separately'''

    common_term = (2.0 / num_values) * err #common term=dc/derr

    d_err_d_b_o = 2 + (lambda1 * 2 * x) + (mu1 * x**2)#this calculates the rate of change of error wrt output bias b_o 
    grad_b_o = np.sum(common_term * d_err_d_b_o)#this calculates the gradient of the cost function wrt to output bias b_o

    d_err_d_v = (a_h * (2 + 2*lambda1*x + mu1*x**2)) + \
                (sig_prime * w_h * (4*x + lambda1*x**2)) + \
                (sig_double_prime * (w_h**2) * x**2)
    grad_v = np.sum(common_term * d_err_d_v, axis=1)#this calculates the gradient of the cost function wrt to output weights v

    term1 = sig_prime * (2 + 2*lambda1*x + mu1*x**2)
    term2 = sig_double_prime * w_h * (4*x + lambda1*x**2)
    term3 = sig_triple_prime * (w_h**2) * (x**2)
    d_err_d_b_h = v.T * (term1 + term2 + term3)#this calculates the rate of change of error wrt hidden bias b_h
    grad_b_h = np.sum(common_term * d_err_d_b_h, axis=1)#this calculates the gradient of the cost function wrt to hidden bias b_h

    term1_w = sig_prime * (6*x + 3*lambda1*x**2 + mu1*x**3)
    term2_w = sig_double_prime * w_h * (6*x**2 + lambda1*x**3)
    term3_w = sig_triple_prime * (w_h**2) * (x**3)
    d_err_d_w_h = v.T * (term1_w + term2_w + term3_w)#this calculates the rate of change of error wrt hidden weights 
    grad_w_h = np.sum(common_term * d_err_d_w_h, axis=1)#this calculates the gradient of the cost function wrt to hidden weights w_h


    grad_P0 = np.stack((grad_b_h, grad_w_h), axis=-1)
    grad_P1 = np.concatenate(([grad_b_o], grad_v)).reshape(1, -1)
    
    return [grad_P0, grad_P1]#returns the desired value of (dc/dv,dc/db_o,dc/dw_h,dc/db_h)
'''this section of code is used to calculate the solution analytically which can be used to compare the performance of our network'''
def g_analytic_2nd_order(x, lambda1=5, mu1=6, g_0=2, g_0_dash=1):
    """
    Solves g'' + lambda1*g' + mu1*g = 0
    Handles all three cases: over-damped, critically-damped, and under-damped.
    """
    discriminant = (lambda1**2) - (4*mu1)#calculated the discriminant b^2-4ac
    
    if discriminant < 0: #underdamped case
        # Under-damped (complex roots)
        a = -lambda1 / 2.0
        b = np.sqrt(4*mu1 - lambda1**2) / 2.0
        C1 = g_0
        C2 = (g_0_dash - a*C1) / b
        return np.exp(a*x) * (C1 * np.cos(b*x) + C2 * np.sin(b*x))
    
    elif discriminant == 0: #critically damped case
        r = -lambda1 / 2.0
        C1 = g_0
        C2 = g_0_dash - r * g_0
        return (C1 + C2 * x) * np.exp(r * x)
    
    else:
        # Over-damped (real roots)
        r1 = (-lambda1 + np.sqrt(discriminant)) / 2.0
        r2 = (-lambda1 - np.sqrt(discriminant)) / 2.0
        C1 = (g_0_dash - r2*g_0) / (r1 - r2)
        C2 = g_0 - C1
        return (C1*np.exp(r1*x)) + (C2*np.exp(r2*x))
    
'''Part 5:'''

'''solver function '''

'''This section contains the function which takes the input such as number of iterations,
                            parameters, input vectors etc, and tries to learn the solution
                            This section also asks which degree of ode we want to solve,takes the coefficients of the differential terms and 
                            intial condition for our system.'''
def solve_ode_neural_network(x, num_neurons_hidden, num_iter, lmb,
                             ODE_ORDER, **kwargs):
    
# ---Scale initial weights for stability ---
    p0 = npr.randn(num_neurons_hidden, 2 ) * 0.1#generates a random set of weights and biases for the hidden layer 
    p1 = npr.randn(1, num_neurons_hidden + 1 ) * 0.1#generates a random set of weights and biases for the output layer
    P = [p0, p1]

    # Print initial cost
    if ODE_ORDER == '1st':
        print('Initial cost (1st order): %g' % cost_function_1st_order(P, x, kwargs['g0'], kwargs['gamma']))
    else:
        print('Initial cost (2nd order): %g' % cost_function_2nd_order(P, x, kwargs['lambda1'], kwargs['mu1'], kwargs['g0'], kwargs['g0_dash']))

    # Training loop
    for i in range(num_iter):
        
        # Call the correct gradient function
        if ODE_ORDER == '1st':
            cost_grad = get_gradients_1st_order(P, x, kwargs['g0'], kwargs['gamma'])
        else:
            cost_grad = get_gradients_2nd_order(P, x, kwargs['lambda1'], kwargs['mu1'], kwargs['g0'], kwargs['g0_dash'])

        # Update parameters
        """lmb is a hyperparameter know as learning rate it tell our neural network how fast to learn from the data or how large of a step to take during gradient descent
         if it is to small the model doesn't learn at all and if its too large the model overshoots the minima and bounces all over the loss function space """
        P[0] = P[0] - lmb * cost_grad[0]#updating the parameters of the hidden layers 
        P[1] = P[1] - lmb * cost_grad[1]#updating the parameters of the output layers
        
        # Print the  cost or the loss after evry 1/10 the of the total number of iterations just to keep track of the loss functions ideally it should be tending to zero
        if (i+1) % (num_iter // 10) == 0:
            if ODE_ORDER == '1st':
                cost = cost_function_1st_order(P, x, kwargs['g0'], kwargs['gamma'])
            else:
                cost = cost_function_2nd_order(P, x, kwargs['lambda1'], kwargs['mu1'], kwargs['g0'], kwargs['g0_dash'])
            print(f"Iteration {i+1}, Cost: {cost:.4e}")

    # Print final cost
    if ODE_ORDER == '1st':
        print('Final cost (1st order): %g' % cost_function_1st_order(P, x, kwargs['g0'], kwargs['gamma']))
    else:
        print('Final cost (2nd order): %g' % cost_function_2nd_order(P, x, kwargs['lambda1'], kwargs['mu1'], kwargs['g0'], kwargs['g0_dash']))

    return P

# --- MAIN EXECUTION ---
'''Part : 6'''
if __name__ == '__main__':
    npr.seed(15)#every time we run our code , the initial "random" weights p0 and p1 are remains same the same, hence it is used to generate eliminate randomness or make the genrator pseudo random
    # Set to '1st' to solve the 1st-order ODE
    # Set to '2nd' to solve the 2nd-order ODE
    ODE_ORDER = input("enter order 1st or 2nd : ")#takes the input as which degree equation we wwant to solve
    # ---!!!--------------------!!! ---

    if ODE_ORDER == '1st':
        # --- 1st Order Problem ---
        print("--- Solving 1st Order ODE: g' + gamma*g = 0 ---")
        
        # --- Parameters ---
        g0 = float(input("enter the intial condition value y(0): "))
        gamma = float(input("enter the gamma value"))
        N = 10#this is the number of data points on which the model will be trained
        x = np.linspace(0, 1, N)#generates the input vector x
        num_hidden_neurons = 10#defines the number of hidden neurons
        num_iter = 20000#it is the number of iterations for which we train our model
        lmb = 0.01#defines the learning rate
        
        # --- Solve ---
        P = solve_ode_neural_network(x, num_hidden_neurons, num_iter, lmb,
                                     ODE_ORDER, g0=g0, gamma=gamma)

        # --- Plotting ---
        '''this section of code is just plotting our predicted and expected results using the output of the neural network and the analytic function which we predefined'''
        x_high_res = np.linspace(0, 1, 100)
        res_analytical = g_analytic_1st_order(x_high_res, gamma, g0)
        res_nn = g_trial_1st_order(x, P, g0)
        
        print('Max absolute difference: %g' % np.max(np.abs(res_nn[0,:] - g_analytic_1st_order(x, gamma, g0))))
        
        plt.figure(figsize=(10,6))#sets the dimension of the figure
        plt.title(f"1st Order: $g' + {gamma}g = 0$ (Gradients)")#sets the title for the plot
        plt.plot(x_high_res, res_analytical, label='Analytical', linewidth=2, color='blue')#plots the analytical function
        plt.plot(x, res_nn[0,:], label='predicted Solution', linestyle='--', marker='o', color='red')#plots the predicted function
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('g(x)')
        
    else:
        # --- 2nd Order Problem ---
        print("--- Solving 2nd Order ODE: g'' + lambda*g' + mu*g = 0 ---")
        
        # --- Parameters ---
        lambda1 = float(input("enter the coefficient of dy_dx"))
        mu1 = float(input("enter the coefficient of y"))
        g0 = float(input("enter the intial condition value of y at x or t=0:"))
        g0_dash = float(input("enter the intial condition value of dy_dx at t=0:"))
        N = 30#this is the number of data points on which the model will be trained
        x = np.linspace(0, 1, N)#generates the input vector x
        num_hidden_neurons = 10#defines the number of hidden neurons
        num_iter = 80000#it is the number of iterations for which we train our model
        lmb = 0.001#defines the learning rate

        # --- Solve ---
        P = solve_ode_neural_network(x, num_hidden_neurons, num_iter, lmb,
                                     ODE_ORDER, lambda1=lambda1, mu1=mu1, g0=g0, g0_dash=g0_dash)

        # --- Plotting ---
        '''this section of code is just plotting our predicted and expected results using the output of the neural network and the analytic function which we predefined'''
        x_high_res = np.linspace(0, 1, 100)
        res_analytical = g_analytic_2nd_order(x_high_res, lambda1, mu1, g0, g0_dash)
        res_nn = g_trial_2nd_order(x, P, g0, g0_dash)
        
        print('Max absolute difference: %g' % np.max(np.abs(res_nn[0,:] - g_analytic_2nd_order(x, lambda1, mu1, g0, g0_dash))))
        
        plt.figure(figsize=(10,6))#sets the dimension of the plot
        plt.title(f"2nd Order: g'' + {lambda1}g' + {mu1}g = 0")#sets th title of the plot
        plt.plot(x_high_res, res_analytical, label='Analytical', linewidth=2, color='blue')#plots the analytical solution
        plt.plot(x, res_nn[0,:], label='predicted Solution', linestyle='--', marker='o', color='red')#plots the predicted solution
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('g(x)')
    plt.show()#shows th output
