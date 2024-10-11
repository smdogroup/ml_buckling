# ML_BUCKLING
## Machine Learning to Improve Buckling Predictions for Efficient Structural Optimization of Stiffened Structures
ML_BUCKLING uses Gaussian process (GP) surrogate models to predict panel buckling loads based on training data from high-fidelity finite element buckling analysis. These predicted panel buckling loads are designed for structural optimization applications. ML_BUCKLING provides routines for the  training of the GP surrogate models and the parametric studies of high-fidelity finite element buckling analysis. 



## How to Cite Us
Please cite the following paper.

Sean P. Engelstad, Brian J. Burke and Graeme J. Kennedy. "Machine Learning to Improve Buckling Predictions for Efficient Structural Optimization of Aircraft Wings," AIAA 2024-3981. AIAA AVIATION FORUM AND ASCEND 2024 . July 2024. [https://arc.aiaa.org/doi/abs/10.2514/6.2024-3981](https://arc.aiaa.org/doi/abs/10.2514/6.2024-3981)

## Installation
Clone the repo and run one of the following commands in the main folder.

1. For regular users of the code:
`pip install .`
2. To obtain an editable installation for contributing to our code use:
`pip install -e .`

## DEPENDENCIES for the Structural Optimization Examples
The example structural optimizations `4/aob_opt` and `5/hsct_opt` in this repo use the following dependencies.
- [TACS](https://github.com/smdogroup/tacs)
- [FUNtoFEM](https://github.com/smdogroup/funtofem)
- [PyOptSparse](https://github.com/mdolab/pyoptsparse)
