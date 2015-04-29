l = 5e5;
Point(1) = {-l, -l, 0};
Point(2) = {l, -l, 0};
Point(3) = {l, l, 0};
Point(4) = {-l, l, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {3, 4, 1, 2};

Field[1] = MathEval;
Field[1].F = "0.5e5";
Background Field = 1;

//Field[1] = MathEval;
//Field[1].F = "0.5e4+1*(abs(x)+abs(y))";
//Background Field = 1;

Plane Surface(6) = {5};
Physical Surface(1)={6};
Physical Line(1) = {1,2,3,4};
//Physical Surface("Surface")={6}; // firedrake does not support strings
//Physical Line("Wall") = {1,2,3,4};

Mesh.CharacteristicLengthExtendFromBoundary=1;
Mesh.CharacteristicLengthFromPoints=1;
Mesh.SecondOrderIncomplete=0;

