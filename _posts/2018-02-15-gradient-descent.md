---
layout: post
title: "Gradient Descent"
date: 2018-02-15 11:33:47
image: '/assets/img/'
description: 'An optimization algorithm to find a minimum of a function'
tags:
- gradient descent
- optimization
- neural nets
- stochastic
- mini-batch
categories:
- Neural Networks
twitter_text: 
---

# Gradient Descent
Gradient descent is an iterative optimization algorithm for findinig the minimum of a function. The basic idea is, we have a cost function that we want to minimize. We start from a point calculate the negative gradient and move further in that direction, eventually reaching the minimum.

![Output](/assets/img/NeuralNets/1.png){:class="img-responsive"}

## GD with Linear Regression
- Say we have two parameters affecting the cost function (β0,β1)
- This creates a surface as shown below rather than a line as above
- Now how can we find the minimum without knowing how the surface J(β0,β1) looks like?

**Well here's what we can do:**

- First we compute the gradient ∇J(β0,β1), which will point us to the direction of the biggest increase
- So we know -∇J(β0,β1) points in the opposite direction(the direction we want)

![Output](/assets/img/NeuralNets/2.png){:class="img-responsive"}

The gradient is actually a vector which contains co-ordinates of the partial derivaties of the parameters shown in the equation below.
![Output](/assets/img/NeuralNets/3.png){:class="img-responsive"}

We can therefore use the gradient(∇) and the cost function to calculate the next point from the current point to move towards(ω1 -> ω2)
![Output](/assets/img/NeuralNets/4.png){:class="img-responsive"}

We do this iteratively until we get to the minimum. α is used to choose how much of a jump we make from ω1 -> ω2 etc. Too little of a jump means we have to iterative many times. Too big of a jump means we will overestimate and possibly miss the minima.

![Output](/assets/img/NeuralNets/5.png){:class="img-responsive"}

## Stochastic Gradient Descent
Instead of using all the data, you can use just one data point to determine a gradient and cost function. But the path may not be most efficient, caused by noise from a single point. Also on a high feature space it may not obtain the most effective minima.
![Output](/assets/img/NeuralNets/6.png){:class="img-responsive"}

## Mini Batch Gradient Descent
There is a third way to perform gradient descent. It involves performing an update for every n training examples. This is the `best of both worlds` because:
- Less memory usage compared with normal gradient descent
- Less noisy than stochastic gradient descent

![Output](/assets/img/NeuralNets/7.png){:class="img-responsive"}
