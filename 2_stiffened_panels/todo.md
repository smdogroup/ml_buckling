# TODO in Stiffened Panels

* axial dataset and surrogate model are good now after deep investigation into dataset!
* currently shear dataset was ran and surrogate model resulted in gamma linear term going to zero
    - [ ] describe in paper that if gamma linear coeff goes to zero, indicative of bad dataset and deep investigation into some troublesome models and consistency of solves had to be done
    - [ ] add to paper that RQ coeff and gamma coeff going to ub and lb respectively is sign of bad or inconsistent data
- [ ] investigate individual shear models with high FEA/CF ratio
    - [ ] make script print out rho0, xi, gamma, N_s, etc. values for high FEA/CF
    - [ ] make unittests that investigate these shear models
    - [ ] improve ant optimization or sigma or error check => try to get these models more consistent
- [ ] then retrain shear model and hope to see gamma, RQ terms reasonable OMAGs