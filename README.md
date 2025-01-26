
## Overview
This repository contains the code for the experiments performed in the submitted paper.
  
Contents of this repository:
- **README** This file.
- **AAAI2025_zeroth-order_experiment.py** The Python script used in our experiments.
- **data** This directory contains the datasets used in our experiments.

## Requirements
The code was implemented in Python 3.6.8.

## Usage
1. To run the experiment, execute the following command ``python3 AAAI2025_zeroth-order_experiment.py 0.19 0.0001 0.001 0.001 2 0.1 10 5000 X 10 40 0.5 20", where X should be replaced with the desired data ID.
   The available data IDs are: 8, 12, 21, 25, 29, 32, 38, 49. Each data ID corresponds to a specific week of 2022.
2. The results will be saved in a folder named according to the input arguments.
  
If you wish to set the parameters manually, use the following command "python3 AAAI2025_zeroth-order_experiment.py mu_0 mu_min mu_CZO beta_0 m_k_coefficient M s_max max_samples data_ID location_initial_points number_o simulations"
The description for each argument is as follows:
mu_0: Parameter \mu_0 in the Proposed-1 method.
mu_min: Parameter \mu_{\min} in the Proposed-1 method.
mu_CZO: Parameter \mu used to estimate the gradient in the CZO method.
beta_0: The initial step-size of the Proposed-1 method.
m_k_coefficient: The coefficient for increasing the mini-batch size relative to the number of iterations.
M: Parameter M in the Proposed-1 method.
s_max: Parameter s_max in the Proposed-1 method.
max_samples: The maximum number of samples used as the termination condition.
data_ID: The ID of the data to be used
location_initial_points: This value, multiplied by a vector with all elements equal to 1, is used as the initial point.
number_of_simulations: The number of times the experiment is repeated.


