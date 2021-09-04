L = 19000; // domain length
H = 7000;  // domain height
l = 4000;  // distance before cylinder

R = 500;   // cylinder radius
X = 0;     // cylinder center
Y = 0;
lc = 100;

Point(1) = {0-X,0-Y,0,lc};
Point(2) = {R-X,0-Y,0,lc};
Point(3) = {0-X,R-Y,0,lc};
Point(4) = {0-X,-R-Y,0,lc};
Point(5) = {-R-X,0-Y,0,lc};
Point(6) = {-l-X,-H/2-Y,0,lc};
Point(7) = {-l-X,H/2-Y,0,lc};
Point(8) = {-l+L-X,H/2-Y,0,lc};
Point(9) = {-l+L-X,-H/2-Y,0,lc};
Point(10) = {-l+L*0.9-X,0-Y,0,lc};

Circle(1) = {3,1,2};
Circle(2) = {2,1,4};
Circle(3) = {4,1,5};
Circle(4) = {5,1,3};
Line(5) = {7,8};
Line(6) = {8,9};
Line(7) = {9,6};
Line(8) = {6,7};
Line(9) = {2,10};
Line Loop(9) = {7,8,5,6};
Line Loop(10) = {3,4,1,2};
Plane Surface(11) = {9,10};

Compound Curve {1,2,3,4};
Compound Surface {11};

Physical Surface(10) = {11};
Physical Line(1) = {8};
Physical Line(2) = {6};
Physical Line(3) = {7};
Physical Line(4) = {5};
Physical Line(5) = {1,2,3,4};

Field[1] = Attractor;
Field[1].CurvesList = {1, 2, 3, 4};
Field[1].NumPointsPerCurve = 100;
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].DistMax = 7000;
Field[2].DistMin = 1000;
Field[2].SizeMax = 100;
Field[2].SizeMin = 30;
Field[3] = MathEval;
Field[3].F = "F2^1.6";

Field[4] = Attractor;
Field[4].CurvesList = {9};
Field[4].NumPointsPerCurve = 100;
Field[5] = Threshold;
Field[5].InField = 4;
Field[5].DistMax = 10000;
Field[5].DistMin = 1500;
Field[5].SizeMax = 100;
Field[5].SizeMin = 30;
Field[6] = MathEval;
Field[6].F = "F5^1.6";

Field[7] = Min;
Field[7].FieldsList = {3, 6};
Background Field = 7;

// mesh size is fully determined by the background field
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

// 1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 5: Delaunay,
// 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay for Quads,
// 9: Packing of Parallelograms
Mesh.Algorithm = 6;
