// Straight channel for smagorinsky_channel.py.
//
// The array region is a separate elementary surface so that it is retained as
// Physical Surface 100 ("array_refinement") in the generated .msh file.
// The size field refines this region, with a 30 m transition outside it.

SetFactory("OpenCASCADE");

channel_length = 850.0;
channel_width = 300.0;

// Encloses the centred two-by-two turbine array, with a 2.5D margin around
// the outer turbines (D = 20 m).  The downstream wakes use the coarser mesh.
array_xmin = 300.0;
array_xmax = 550.0;
array_ymin = 50.0;
array_ymax = 250.0;

lc_coarse = 20.0;
lc_array = 4.0;

// Exterior channel boundary, counter-clockwise.
Point(1) = {0, 0, 0, lc_coarse};
Point(2) = {channel_length, 0, 0, lc_coarse};
Point(3) = {channel_length, channel_width, 0, lc_coarse};
Point(4) = {0, channel_width, 0, lc_coarse};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Array-region boundary, counter-clockwise.
Point(11) = {array_xmin, array_ymin, 0, lc_array};
Point(12) = {array_xmax, array_ymin, 0, lc_array};
Point(13) = {array_xmax, array_ymax, 0, lc_array};
Point(14) = {array_xmin, array_ymax, 0, lc_array};
Line(11) = {11, 12};
Line(12) = {12, 13};
Line(13) = {13, 14};
Line(14) = {14, 11};

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {11, 12, 13, 14};
Plane Surface(1) = {1, 2};
Plane Surface(100) = {2};

// Keep the channel boundary IDs used by smagorinsky_channel.py.
Physical Curve("inflow", 1) = {4};
Physical Curve("outflow", 2) = {2};
Physical Curve("south_wall", 3) = {1};
Physical Curve("north_wall", 4) = {3};
Physical Surface("channel", 1) = {1};
Physical Surface("array_refinement", 100) = {100};

Field[1] = Box;
Field[1].VIn = lc_array;
Field[1].VOut = lc_coarse;
Field[1].XMin = array_xmin;
Field[1].XMax = array_xmax;
Field[1].YMin = array_ymin;
Field[1].YMax = array_ymax;
Field[1].Thickness = 30.0;
Background Field = 1;

