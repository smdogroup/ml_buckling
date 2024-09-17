# Structural Optimization of a Benchmark Tutorial Wing
## By: Sean Engelstad, Brian Burke
## Original source of the benchmark wing: Alasdair Gray

* The closed-form and GP model approaches are compared on the structural optimization of a single stiffened panel. FUNtoFEM and TACS are used to handle the oneway structural optimization (in this case of course with no aero loads at all, just prescribed struct loads).
* This is a simple debugging optimization case for stiffened panel structural optimization. The goal is to see how to reduce the number of optimization iterations for doing so.
* Note: in the TACS output *.f5 files. Convert them to vtk or tecplot format. Then, the dv1-dv7 stand for the following for these constitutive objects.
dv1 - effective thickness
dv2 - effective bending thickness
dv3 - panel length
dv4 - stiffener pitch
dv5 - panel thickness
dv6 - stiffener height
dv7 - stiffener thickness

(note dv8 for panel width is not shown nominally in TACS, would have to write this out yourself by recompiling TACS and changing the source code.)

## Reducing the Number of Optimization Iterations to Convergence
* Can't eliminate penalty parameter here.. then it steps too far into the infeasible region. Explore other ways to reduce # of optimization steps.