basin_x = 640;
basin_y = 320;
site_x = 320;
site_y = 160;
element_size = 2.5;
element_size_coarse = 10;

Point(1) = {0, 0, 0, element_size_coarse};
Point(2) = {basin_x, 0, 0, element_size_coarse};
Point(3) = {0, basin_y, 0, element_size_coarse};
Point(4) = {basin_x, basin_y, 0, element_size_coarse};

Point(5) = {(basin_x - site_x)/2, (basin_y - site_y)/2, 0, element_size};
Extrude{site_x, 0, 0} { Point{5}; Layers{site_x/element_size}; }
Extrude{0, site_y, 0} { Line{1}; Layers{site_y/element_size}; }

Line(6) = {1, 2};
Line(7) = {2, 4};
Line(8) = {4, 3};
Line(9) = {3, 1};
Line Loop(10) = {9, 6, 7, 8};
Line Loop(11) = {3, 2, -4, -1};
Plane Surface(12) = {10, 11};
Physical Surface(0) = {12};
Physical Surface(2) = {5};
Physical Line(2) = {7};
Physical Line(1) = {9};
Physical Line(3) = {8, 6};
