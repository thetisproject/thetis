//*********** lockExchange mesh *************//
Cx = 32000;

// lc = 125;  // fine, 160 layers
lc = 250; // medium2, 80 layers
//lc = 500; // medium, 40 layers
// lc = 2000; // coarse, 10 layers
Cy = 0.82*lc;

Point(1) = {-Cx, -Cy, 0, lc};
Point(2) = {Cx , -Cy, 0, lc};
Point(3) = {Cx , Cy , 0, lc};
Point(4) = {-Cx, Cy , 0, lc};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Physical Line(1) = {1,3};
Physical Line(2) = {2};
Physical Line(3) = {4};

Line Loop(5) = {1,2,3,4};
Plane Surface(6) = {5};
Physical Surface(11) = {6};

Mesh.Algorithm = 6; // frontal=6, delannay=5, meshadapt=1
