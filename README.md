# Solving-Differential-equation-with-Neural-Nets

The repo contains very simple implementation of solving a differential equations with the help of neural network(Physics Informed Neural Networks(PINNs)).
Because whatever physical property we want our system,we model a differential equation for that property and then implement that as a loss function
for our network. 
There are many other methods to incorporate physics and symmetry related to physics such as by augmenting the data or choosing an architecture that forces that physics by construction, one of the method is to craft a lost function using our differential equation and let our plain neural network learn.

Mostly the models are trained using deeplearning libraries such as tensorflow, pytorch etc. which makes this neural networks kind of black box. Also they use automatic differentiation to calculate the gradients of the loss function with respect to the parameters and inputs,which is necessary for higher order Odes and Pdes.

Although the method of not using library makes our code slow as compared to the optimized matrix calculus but it lets us see how the networks learn
