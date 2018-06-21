W=200; // width of channel
L=1e3; // length of channel
dx1=10;
dx2=2;
D=18.; // turbine diameter
xt=50.; // x location of turbine
yt=W/2; // y location of turbine
Point(1) = {0., 0., 0., dx1};
Point(2) = {L, 0., 0., dx1};
Point(3) = {L, W, 0., dx1};
Point(4) = {0., W, 0., dx1};
Point(5) = {xt-D/2, yt-D/2, 0., dx2};
Point(6) = {xt+D/2, yt-D/2, 0., dx2};
Point(7) = {xt+D/2, yt+D/2, 0., dx2};
Point(8) = {xt-D/2, yt+D/2, 0., dx2};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Physical Line(1) = {4}; // Left boundary
Physical Line(2) = {2}; // Right boundary
Physical Line(3) = {1,3}; // Sides
// outside loop
Line Loop(1) = {1, 2, 3, 4};
// inside loop
Line Loop(2) = {5, 6, 7, 8};
Plane Surface(1) = {1,2};
Plane Surface(2) = {2};
// id outside turbine
Physical Surface(1) = {1};
// id inside turbine
Physical Surface(2) = {2};
