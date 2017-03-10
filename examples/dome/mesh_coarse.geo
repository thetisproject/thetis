// ***** mesh for DOME test case ***** //
Lx = 1100e3;
Ly = 600e3;
Lriver = 200e3;
Wriver = 100e3;
Xriver = 800e3;
Yslope = 300e3;
Xextend = 120e3;
lc = 1e20;
// typical resolution dx=dy=10 km, 21 sigma levels

Point(1) = {0, 0, 0, lc};
Point(2) = {0, Yslope, 0, lc};
Point(3) = {0 , Ly, 0, lc};
Point(4) = {Xriver, Ly , 0, lc};
Point(5) = {Xriver, Ly+Lriver , 0, lc};
Point(6) = {Xriver+Wriver, Ly+Lriver , 0, lc};
Point(7) = {Xriver+Wriver, Ly, 0, lc};
Point(8) = {Lx, Ly, 0, lc};
Point(9) = {Lx, Yslope, 0, lc};
Point(10) = {Lx, 0, 0, lc};
Point(11) = {-Xextend, 0, 0, lc};
Point(12) = {-Xextend, Yslope, 0, lc};
Point(13) = {-Xextend, Ly, 0, lc};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,9};
Line(9) = {9,10};
Line(10) = {10,1};

Line(11) = {2, 9};
Line(12) = {4, 7};

Line(13) = {11, 12};
Line(14) = {12, 13};
Line(15) = {1, 11};
Line(16) = {2, 12};
Line(17) = {3, 13};

Physical Line(1) = {13,14};    // west
Physical Line(2) = {8,9};      // east
Physical Line(3) = {10,15};    // south
Physical Line(4) = {5};        // river inlet
Physical Line(5) = {-17,3,4,6,7};  // rest; closed boundary

// lower basin
Line Loop(5) = {15,13,-16,11,9,10};
Plane Surface(6) = {5};
// Physical Surface(7) = {6};

// upper basin
Line Loop(6) = {16,14,-17,3,12,7,8,-11};
Plane Surface(7) = {6};
// Physical Surface(8) = {7};

// embayment
Line Loop(7) = {4,5,6,-12};
Plane Surface(8) = {7};
// Physical Surface(9) = {8};

Physical Surface(9) = {6,7,8};

Field[1] = MathEval;  // min reso
Field[1].F = "20e3";
Field[2] = MathEval;  // max reso
Field[2].F = "40e3";
Field[3] = MathEval;  // y ramp
Field[3].F = "F2 + (-y + 300e3)/300e3*(F2 - F1)";
Field[4] = Max;
Field[4].FieldsList = {1, 3};
Field[5] = Min;
Field[5].FieldsList = {2, 4};
Background Field = 5;

Mesh.Algorithm = 6; // frontal=6, delannay=5, meshadapt=1
