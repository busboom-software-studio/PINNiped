# Physics Informed Neural Networks

This project will explore using Neural Networks to solve systems of ordinary
and partial differential equations, with a specific focus on those that
represent physical systems.  The main goal of the project will be to
accurately model balistics with a complex projectile, specifically the [note](
https://www.andymark.com/products/frc-2024-am-4999) 
used in the [2024 FIRST Robotics game, CRESCENDO. ](https://firstfrc.blob.core.windows.net/frc2024/Manual/2024GameManual.pdf)

The interesting problem in the CRESCENDO game is that the projectile, [a foam
donut called a 'note'](https://www.andymark.com/products/frc-2024-am-4999) is
often launched with spin to increase sability. But this gyroscopic stability
will also mean that the angle of the velocity vector relative to the
orientatino of the projectile will change over the course of its trajectory,
and this will result in a change in drag. Additionally, there may be other
forces induced by the motion of the robot launching the projective. So, for
this project, we will develop neural networks that can predict trajectories
with complex drag functions, based both on simulations, and on data collected
from the robots. 


We will begin by studying:

* Ballistics equations
* Calculating systems of ODEs
* Basic dense neural networks
* Basic machine vision processing

Once students are comforrtable with the basics, we will move on to:

* Modeling systems with Recurrent and LSTM neural networks
* Developiong a physics informed neural network. 


The physics informed network is very likely to be a combination of the
traditional blassitics equations with the addition of a learned function for
drag, however, we might discover that the drag function can be learned and
predicted through other machine learning techniques. 


## Getting Started

Start with this [basic introduction to neural networks](https://www.knime.com/blog/a-friendly-introduction-to-deep-neural-networks)


Be familiar with the [ballistics equations and how to solve them in python](https://blog.finxter.com/5-best-ways-to-model-projectile-motion-using-python/). See also this more 
[sophisticated solution using scipy. ](https://scipython.com/book2/chapter-8-scipy/examples/a-projectile-with-air-resistance/)


This video illustrates how a NN can be a [universal approximator of continuous functions. ](https://x.com/Hamptonism/status/1796111788292866468). The gist of the video is that for each connection, the weight of the connection is setting the slope of the RELU activation function, and the bias is shifting it. 


## Notes

[NeuroDiffEq](https://github.com/NeuroDiffGym/neurodiffeq) A Python package for solving differential equations with neural networks.



A good [introduction to PINNs](https://medium.com/@theo.wolf/physics-informed-neural-networks-a-simple-tutorial-with-pytorch-f28a890b874a)

Videos: 
* Overview of PINNS:  https://www.youtube.com/watch?v=qYmkUXH7TCY
* Solving NS with as PINN, with code: https://www.youtube.com/watch?v=ISp-hq6AH3Q

Repo with a PINN that solves [Navier Stokes](https://github.com/hojunkim13/PINNs), and an [associated paper](https://maziarraissi.github.io/PINNs/)

Lots of examples of physics simulations in Javascript: https://matthias-research.github.io/pages/tenMinutePhysics/index.html

Extensive [video lectures](https://www.youtube.com/@Eigensteve) on physics and AI. 

Here are several resources that can help you understand and implement a physics-informed neural network (PINN) for solving ballistics problems:

Hands-on Introduction to Physics-Informed Neural Networks: This tutorial on nanoHUB offers a comprehensive introduction to PINNs, focusing on how to incorporate differential equations into neural networks. It includes practical examples implemented in PyTorch, making it suitable for users familiar with conventional neural network training and PyTorch basics​ ("NanoHUB")​.

https://nanohub.org/resources/handsonpinns

TensorFlow Tutorial for PINNs: This tutorial provides a step-by-step guide to implementing PINNs using TensorFlow. It explains the core concepts of PINNs, including how to incorporate partial differential equations (PDEs) into the loss function, and provides code examples that you can adapt for your ballistics application​ (George Miloshevich)​.

https://georgemilosh.github.io/blog/2022/distill/

GitHub Repository by Maziar Raissi: This repository contains extensive resources and examples on using PINNs for solving forward and inverse problems involving nonlinear PDEs. It includes detailed documentation and several example projects that demonstrate the application of PINNs to different types of physical problems, which can be highly relevant for your ballistics project​ (GitHub)​.

https://github.com/maziarraissi/PINNs




