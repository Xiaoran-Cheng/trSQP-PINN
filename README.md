# Physics-Informed Neural Networks with Trust-Region Sequential Quadratic Programming
## Introduction
PINN is a recent development in the Scientific ML field that incorporates physical information into the loss function. We delve deeper into the challenges and solutions for using PINN in solving complex PDEs. This project highlights the optimization issues faced by PINN, particularly in intricate scenarios, and introduces advanced hard-constrained optimization methods like the $L_2^2$ Penalty Method, Augmented Lagrangian Method, and $\textbf{Trust Region Sequential Quadratic Programming}$. These methods are shown to significantly enhance the accuracy of solutions. The research includes detailed experiments on three different PDEs: transport equation, reaction equation, and reaction-diffusion equation, demonstrating the effectiveness of these methods in various contexts.

## Usage
For example, to run experiments for transport equations
```python
main_transport.py
```
