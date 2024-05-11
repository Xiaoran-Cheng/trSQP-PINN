# Trust-Region Sequential Quadratic Programming for Physics-Informed Neural Networks (trSQP-PINN)

## Introduction
Physics-Informed Neural Networks (PINNs) are a novel development in the field of Scientific Machine Learning, integrating physical domain knowledge into the learning process through a soft penalized loss function. We introduce a hard-constrained deep learning method, trust-region Sequential Quadratic Programming (trSQP-PINN), designed to overcome the common failure modes of PINNs in solving partial differential equations (PDEs). Our comprehensive experiments demonstrate trSQP-PINN's superior performance compared to traditional PINNs and other hard-constrained methods such as penalty methods and augmented Lagrangian methods applied to PINNs.

## Usage
To run experiments for transport equations using trSQP-PINN, modify the "system" parameter in `main.py` as needed. Execute the following command to start the experiments:
```bash
python main.py
