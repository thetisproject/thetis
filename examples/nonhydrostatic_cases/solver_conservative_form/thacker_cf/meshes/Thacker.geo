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
//Field[1].F = "100" // Coarse
//Field[1].F = "((x-4.521)*(x-4.521)+(y-1.696)*(y-1.696)^0.5)/10+0.1"; // Quadratic around gauge 2
//Field[1].F = "((x-4.521)*(x-4.521))^0.5/10+0.1"; // linear refining towards right
//Field[1].F = "((x-4.521)*(x-4.521))^0.5/10+0.1"; // linear refining towards right plus refinment at the island
//Field[1].F = "min(((x-4.521)*(x-4.521))^0.5/10 + 0.1, (((x-3.39)^2+(y-1.659)^2)^0.5)^4/1+0.05)"; // linear refining towards right plus refinment at the island
Field[1].F = "1.58/4.2*(((420000-(x^2+y^2)^(1/2))*(420000-(x^2+y^2)^(1/2)))^(1/2)/5 + 10000)"; // linear refining towards right plus refinment at the island

Background Field = 1;
