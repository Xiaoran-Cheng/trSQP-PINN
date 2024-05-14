# Physics-Informed Neural Networks with Trust-Region Sequential Quadratic Programming (trSQP-PINN)
Xiaoran Cheng, Sen Na

Physics-Informed Neural Networks (PINNs) are novel development in the field of Scientific Machine Learning, integrating physical domain knowledge into the learning process through a soft penalized loss function. We introduce a hard-constrained deep learning method, trust-region Sequential Quadratic Programming (trSQP-PINN), designed to overcome the failure modes of PINNs in solving partial differential equations (PDEs). Our comprehensive experiments demonstrate trSQP-PINN's superior performance compared to traditional PINNs and other hard-constrained methods such as penalty methods and augmented Lagrangian methods.


## Setup
All our code is implemented in Python (ver 3.10.12). Install the enssential packages needed for the code, using
```bash
pip install jax
pip install jaxopt
pip install flax
```

## File functions
Data.py: generating labeled and unlabeled data for pretraning and traning.
NN.py: construct the neural network
System.py: define the PDEs, initial conditions and solutions
Visualization.py: drawing solution heatmaps and error graphs
optim_PINN.py: define loss for PINNs and penalty methods
optim_aug_lag.py: define loss for augmented Lagrangian methods
optim_sqp.py: define loss and optimization for trSQP-PINN
uncons_opt.py: optimization process for PINNs, penalty and augmented Lagrangian methods
pre_train.py: define loss and optimization for pretraning phase
main.py: define problem and algorithm tuning parameters and main excution file

## Usage
To run experiments for transport equations using trSQP-PINN, modify the "system" parameter in `main.py` as needed. Execute the following command to start the experiments:
```bash
python main.py
```
