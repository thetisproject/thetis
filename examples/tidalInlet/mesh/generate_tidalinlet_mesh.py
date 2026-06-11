import gmsh
import sys

gmsh.initialize()
gmsh.model.add("Tidal Inlet")

# Parameters
length_x = 15e3   # 15 km
length_y = 14e3   # 14 km
meshSize = 230    # approximate element size

NumPointsPerCurve = 6000  # arbitrary set, not really necessary

# Wall y-location
y_start = 6600
y_end = 6800

# Wall segments
x_left = 6500
x_right = 8500

HR = True  # high–resolution flag for inlet walls (l3, l9)
hrfactor = 0.3  # factor of meshSize for HR segments
maxdistfactor = 5.0  # factor to multiply meshSize for max distance in Threshold field


# Add points
# --------------------------------
# Define basin corners (counter-clockwise)
p1 = gmsh.model.geo.addPoint(0, 0, 0, meshSize)  # lower left corner
p2 = gmsh.model.geo.addPoint(length_x, 0, 0, meshSize)  # lower right corner

p21 = gmsh.model.geo.addPoint(length_x, y_start, 0, meshSize)  # right side complications
p22 = gmsh.model.geo.addPoint(x_right, y_start, 0, meshSize)
p23 = gmsh.model.geo.addPoint(x_right, y_end, 0, meshSize)
p24 = gmsh.model.geo.addPoint(length_x, y_end, 0, meshSize)

p3 = gmsh.model.geo.addPoint(length_x, length_y, 0, meshSize)  # upper right corner
p4 = gmsh.model.geo.addPoint(0, length_y, 0, meshSize)  # upper left corner

p41 = gmsh.model.geo.addPoint(0, y_end, 0, meshSize)  # left side complications
p42 = gmsh.model.geo.addPoint(x_left, y_end, 0, meshSize)
p43 = gmsh.model.geo.addPoint(x_left, y_start, 0, meshSize)
p44 = gmsh.model.geo.addPoint(0, y_start, 0, meshSize)

# make lines
# --------------------------------
l1 = gmsh.model.geo.addLine(p4, p41)
l2 = gmsh.model.geo.addLine(p41, p42)
l3 = gmsh.model.geo.addLine(p42, p43)
l4 = gmsh.model.geo.addLine(p43, p44)
l5 = gmsh.model.geo.addLine(p44, p1)
l6 = gmsh.model.geo.addLine(p1, p2)
l7 = gmsh.model.geo.addLine(p2, p21)
l8 = gmsh.model.geo.addLine(p21, p22)
l9 = gmsh.model.geo.addLine(p22, p23)
l10 = gmsh.model.geo.addLine(p23, p24)
l11 = gmsh.model.geo.addLine(p24, p3)
l12 = gmsh.model.geo.addLine(p3, p4)  # closes the outer loop - the open boundary line


# Define line groups for physical boundaries
# --------------------------------
openboundary_lines = [l12]
coastline_lines = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11]

# Create ONLY the full domain surface (single closed loop)
# --------------------------------
outer_loop_lines = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]
outer_loop = gmsh.model.geo.addCurveLoop(outer_loop_lines)
domain_surface = gmsh.model.geo.addPlaneSurface([outer_loop])

# physical groups/curves
# --------------------------------
gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(2, [domain_surface], 1)  # Physical Surface
gmsh.model.setPhysicalName(2, 1, "Fluid_Domain")

gmsh.model.addPhysicalGroup(1, openboundary_lines, 2)  # Open boundary
gmsh.model.setPhysicalName(1, 2, "openboundary")

gmsh.model.addPhysicalGroup(1, coastline_lines, 3)
gmsh.model.setPhysicalName(1, 3, "coastline")

# mesh gen
# --------------------------------

# Allow smaller sizes (must be <= desired refined size)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshSize * hrfactor)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshSize)

gmsh.option.setNumber("Mesh.Optimize", 2)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
gmsh.option.setNumber("Mesh.Smoothing", 10)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

if HR:
    # High‑resolution segment similar to coastline_hr_tag logic in rimini.py
    hr_lines = [l3, l9]

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", hr_lines)
    gmsh.model.mesh.field.setNumber(1, "NumPointsPerCurve", NumPointsPerCurve)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", meshSize * hrfactor)   # target half size
    gmsh.model.mesh.field.setNumber(2, "SizeMax", meshSize)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", meshSize * maxdistfactor)       # transition over ~1 base element

    gmsh.model.mesh.field.setAsBackgroundMesh(2)

gmsh.option.setNumber("Mesh.Algorithm", 6)

gmsh.model.mesh.generate(2)


# -------------------------
# Remove unused (0D) points
# -------------------------
# Find point entities not referenced by any line and remove them from the model
points = gmsh.model.getEntities(0)   # list of (0, tag)
lines = gmsh.model.getEntities(1)   # list of (1, tag)

used_point_tags = set()
for (_, ltag) in lines:
    # getBoundary returns a list of (dim, tag) tuples for entities on the boundary
    bnds = gmsh.model.getBoundary([(1, ltag)], oriented=False)
    for (bdim, btag) in bnds:
        if bdim == 0:
            used_point_tags.add(btag)

unused = [(0, ptag) for (_, ptag) in points if ptag not in used_point_tags]

if unused:
    print(f"Removing {len(unused)} unused point(s): {[t for (_, t) in unused]}")
    # Try removing with OCC first; fall back to GEO if needed
    removed = False
    try:
        gmsh.model.occ.remove(unused)
        gmsh.model.occ.synchronize()
        removed = True
    except Exception:
        try:
            gmsh.model.geo.remove(unused)
            gmsh.model.geo.synchronize()
            removed = True
        except Exception as e:
            print("Warning: failed to remove unused points:", e)
    if removed:
        # If geometry changed, update mesh entity mapping (safe to regenerate surface mesh)
        # regenerate mesh nodes/elements to reflect geometry cleanup
        gmsh.model.mesh.generate(2)

# save
gmsh.write("inlet_v1.msh")
# gmsh.write("inlet_v1.vtk")  # for visualization in Paraview

# Verify physical groups
print("\n--- Physical Groups ---")
for dim in [1, 2]:
    groups = gmsh.model.getPhysicalGroups(dim)
    for tag_dim, tag in groups:
        name = gmsh.model.getPhysicalName(tag_dim, tag)
        entities = gmsh.model.getEntitiesForPhysicalGroup(tag_dim, tag)
        print(f"Dim {tag_dim}, Tag {tag}: '{name}' -> entities {entities}")

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()
gmsh.finalize()
