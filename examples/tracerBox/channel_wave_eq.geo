L = 44294.46; // sqrt(9.81*50)*2000 s
n = 25; // nb elements along channel
lc = L/n;
W = 0.82*2*lc;
Point(1) = {0, 0, 0, lc};
Point(2) = {L, 0, 0, lc};
Point(3) = {L, W, 0, lc};
Point(4) = {0, W, 0, lc};
Line(1) = {4, 3};
Line(2) = {3, 2};
Line(3) = {2, 1};
Line(4) = {1, 4};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};
Physical Line(1) = {2};
Physical Line(2) = {4};
Physical Line(3) = {1,3};
Physical Surface(11) = {6};

Mesh.Algorithm = 6; // frontal=6, delannay=5, meshadapt=1

