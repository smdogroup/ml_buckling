## Improved Closed-Form solution development work for pure shear modes
### Sean Engelstad

Results so far seem to indicate that multiplying by sin^2 in x and y directions and adding a phase shift
to the traditional shear mode shapes of w = A * sin(pi(x1 - lam2*x2) / lam1)*sin(pi*x2/b) actually produce
accurate mode shapes against finite aspect ratios. Indeed, the old Timoshenko assumed mode shapes here were wrong
for intermediate and small AR (or any finite AR really) and only approach the true solution at infinite AR.

My modifications to the mode shape seem to indicate it is possible to analytically model the shear mode shapes. 
The current results seem to indicate the following form works:
w = A * sin(pi(x1 - lam2*x2 - 0.5 * lam1)/lam1) * sin^2(pi*x1/lam1) * sin^2(pi*x2/b)
    where lam1 = a / m1 and m1 is varied among m1 = 1,2,3,4,... but is fixed for any given mode shape.
    lam2 is also a continuous variable that is minimized on the critical load N12,cr (and that is how it is determined).
    Earlier versions of this modified mode shape used superellipses - this is also a super-ellipse multiplier (but it's an integer
    so also equivalent to higher period sine wave in x1,x2 direction to help satisfy BCs). Indeed, the crit loads in finite AR
    are much higher than the current shear models - indicating some higher order behavior or restraint by the plate is not accounted for.
    This would account for it!