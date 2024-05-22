# Structural Optimization of a Single Stiffened Panel
## By: Sean Engelstad

* The closed-form and GP model approaches are compared on the structural optimization of a single stiffened panel. FUNtoFEM and TACS are used to handle the oneway structural optimization (in this case of course with no aero loads at all, just prescribed struct loads).
* This is a simple debugging optimization case for stiffened panel structural optimization. The goal is to see how to reduce the number of optimization iterations for doing so.

## Reducing the Number of Optimization Iterations to Convergence
* Ideas: Try to change SNOPT settings. Maybe lower spar pitch (closer to optimal design) will help a lot.