# Physics-Informed Neural Networks with Trust-Region Sequential Quadratic Programming (trSQP-PINN)


## Overview
Physics-Informed Neural Networks (PINNs) represent a cutting-edge advancement in Scientific Machine Learning, integrating physical domain knowledge via a soft penalized loss function. We introduce a novel approach, the **Trust-Region Sequential Quadratic Programming** for PINNs, called **trSQP-PINN**, which addresses the limitations of conventional PINNs in solving Partial Differential Equations (PDEs). Our method enforces hard constraints in the learning process, significantly enhancing model performance. Through extensive experiments, trSQP-PINN demonstrates superior performance compared to standard PINNs and other hard-constrained approaches like penalty and augmented Lagrangian methods applied to PINNs.

## Installation
The following Python packages are required to run the code. Ensure you have Python version 3.10.12 installed and then execute the commands below to install necessary libraries:

```bash
pip install jax jaxopt flax
```

## File Structure
This section details the files included in the project and their specific roles:

- **Data.py**: Responsible for generating both labeled and unlabeled data sets used during the pretraining and training phases.
- **NN.py**: Defines the neural network architecture, setting up the layers and parameters that will model the physical phenomena.
- **System.py**: Contains definitions for the PDEs, initial conditions, and analytic solutions that guide the training process.
- **Visualization.py**: Provides functionality for visualizing the results through solution heatmaps and error graphs, which are crucial for assessing model performance.
- **optim_PINN.py**: Implements the loss functions for standard PINNs and penalty methods, essential for training the models.
- **optim_aug_lag.py**: Specifies the loss functions for the augmented Lagrangian methods.
- **optim_sqp.py**: Manages the loss functions and the optimization process specific to the trSQP-PINN approach.
- **uncons_opt.py**: Manages the general optimization processes for PINNs, penalty methods, and augmented Lagrangian methods.
- **projected_cg.py**: Responsible for performing projected conjugate gradient method. 
- **projection_methods.py**: Helps with defining projection method for projecting gradients onto the constrained space.
- **sensitivity_experiments_graphs.py**: Draw the sensitivity experiments error plots.
- **pre_train.py**: Sets up the loss functions and optimization for the pretraining phase, preparing the neural network for more detailed training.
- **main.py**: Central executable file that configures the problem-specific parameters and algorithm tuning parameters, and launches the main experiments.

Each file is designed to handle specific aspects of the computational process, ensuring that the project is modular and maintainable.


## Usage
To run experiments for transport equations using trSQP-PINN, modify the "system" parameter in `main.py` as needed. Execute the following command to start the experiments:
```bash
python main.py
```
