// Gmsh project created on Wed May 25 13:51:26 2016
Point(1) = {475823.23, 0, 0};
Point(2) = {-475823.23, 0, 0};
Point(3) = {0, 0, 0};
Point(4) = {0, 475823.23, 0};
Point(5) = {0, -475823.23, 0};
Circle(1) = {2, 3, 4};
Circle(2) = {4, 3, 1};
Circle(3) = {1, 3, 5};
Circle(4) = {5, 3, 2};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};
Physical Surface(7) = {6};
Physical Line(8) = {1, 4, 2, 3};

Field[1] = MathEval;
Field[1].F = "1./4.2*(((420000-(x^2+y^2)^(1/2))*(420000-(x^2+y^2)^(1/2)))^(1/2) + 10000)";

Background Field = 1;
