s=5e-1;
s2=1e-1;
L = 15;
W = 10;
k=5;

Point(1) = {0,0,0,s};
Point(2) = {0,W, 0,s};
Point(3) = {L, W, 0,s};
Point(4) = {L, 0, 0,s};

Point(5) = {L/k, W/2, 0,s};


r= 1e0;

Point(6) = {L/k+r, W/2, 0,s2};
Point(7) = {L/k, W/2+r, 0,s2};
Point(8) = {L/k-r, W/2, 0,s2};
Point(9) = {L/k, W/2-r, 0,s2};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Circle(5) = {6,5,7};
Circle(6) = {7,5,8};
Circle(7) = {8,5,9};
Circle(8) = {9,5,6};

Line Loop(1) = {1,2,3,4};
Line Loop(2) = {5,6,7,8};

Plane Surface(1) = {1,2};

Physical Line(1) = 1;
Physical Line(2) = 3;
Physical Line(3) = 4;
Physical Line(4) = 2;
Physical Line(5) = {5,6,7,8};

Physical Surface(1) = 1;


