# Solving-Differential-equation-with-Neural-Nets
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
