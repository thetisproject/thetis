basin_x = 2000;
basin_y = 600;

headland_x_scale = 0.2;
headland_y = 200;

site_x = 160;
site_y = 100;
site_x_start = basin_x/2-site_x/2;
site_x_end = basin_x/2+site_x/2;

site_y_start = basin_y/2 - 50;
site_y_end = site_y_start+site_y;

element_size = 1;
element_size_tip = 10;
element_size_coarse = 50;

Point(1) = {0, 0, 0, element_size_coarse};
Point(2) = {basin_x, 0, 0, element_size_coarse};


// Generate nodes for the headland
res = 100;
For k In {0:res:1}
    x = basin_x/res*k;
    b = 10;
    y = basin_y - headland_y*Exp(-0.5*((headland_x_scale*(x-basin_x/2))/b)^2);
    locres = ((basin_y-y)*element_size_tip + (y-(basin_y-headland_y))*element_size_coarse)/headland_y;
	Point(10+k) = {x, y, 0, locres};
EndFor

// Generate lines for the headland

BSpline(9999) = { 10 : res/2+10 };
BSpline(10000) = { res/2+10 : res+10 };

Line(10001) = {10, 1};
Line(10002) = {1, 2};
Line(10003) = {2, res+10};
Line Loop(10004) = {9999, 10000, -10003, -10002, -10001};

// Generate site nodes
Point(100000) = {site_x_start, site_y_start, 0, element_size};
Extrude{site_x, 0, 0} { Point{100000}; Layers{site_x/element_size}; }
Extrude{0, site_y, 0} { Line{10005}; Layers{site_y/element_size}; }
Line Loop(10010) = {10006, -10008, -10005, 10007};
Plane Surface(10011) = {10004, 10010};
Physical Line(1) = {10001};
Physical Line(2) = {10003};
Physical Line(3) = {10000, 10002};
Physical Surface(1) = {10011};
Physical Surface(2) = {10009};
