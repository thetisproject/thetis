#!/usr/bin/env python

# This script converts selafin files generated for TELEMAC to Gmsh files
# compatible with the Thetis suite of solvers.
# Requirements include the selafin parsers that can be found in the
# telemac source tree. They can be added to your PYTHONPATH using
# $ export PYTHONPATH=$PYTHONPATH:</telemac>/scripts/python27
# You will also need the meshio python module, which can be installed with
# $ pip install meshio


# History
#
# slm06 09/12/2016
# Initial commit

import numpy as np
import sys


def usage():
    print 'USAGE:' 
    print '   slf2gmsh.py <selafin_file> <cli_file>'
    print '   <selafin_file> is the name of the selafin file'
    print '   <cli_file> the name of the cli file'
    print '   Will produce a .msh file with the same name as the input slf'


try:
    import meshio
except ImportError:
    print 'ERROR: Can not import meshio'
    print 'Please try: $ pip install meshio'
    sys.exit(1)


try:
    from parsers import parserSELAFIN
except ImportError:
    print 'ERROR: Can not import selafin parser'
    print 'Have you set your PYTHONPATH correctly?'
    sys.exit(1)


def args_in_list(list_of_args, args):
    return all([True if arg in list_of_args else False for arg in args])


def get_nodes(slfData):
    xCoords = slfData.MESHX
    yCoords = slfData.MESHY
    zCoords = np.zeros(xCoords.shape)    
    return zip(xCoords,yCoords,zCoords)


def get_lines(slfData):
    ''' This returns all lines of the Selafin file. Sometimes this can
    include internal lines, so will have to be filtered using the
    clean_lines function '''
    return parserSELAFIN.getEdgesSLF(slfData.IKLE3, slfData.MESHX, slfData.MESHY)


def clean_lines(lines, clifile):
    ''' Excludes lines that have at least one node not on the boundary '''
    cli_file = open(clifile, 'r')
    boundary_nodes = []
    for line in cli_file:
        boundary_nodes.append(int(line.split()[11]))
    cli_file.close()

    return np.array([line for line in lines if args_in_list(boundary_nodes,[line[0]+1,line[1]+1])])


def write_tmp_mesh(fileName, points, lines, triangles):
    elements = {'line': lines, 'triangle': triangles}
    meshio.write(fileName, points, elements)


def add_tag_to_line(line, tag):
    line.insert(3, str(tag))
    line.insert(3, str(tag))


def add_tags_to_gmsh_from_cli(meshFile, cliFile):
    ''' Gmsh meshes written by meshio have number-of-tags set to 0
    which is incompatible with the DMPlex gmsh reader. This function
    changes the number-of-tags to 2, and adds the following tags:

    1: For all lines on a closed boundary (2 2 2 in the cli file)
    5: For all lines on an open boundary (anything other than 222 in the cli file)
    9: For all triangles (surface ID)

    Suggested future improvement: To distinguish between the different types of
    liquid boundaries.
    '''

    fixed = open(meshFile.replace('_tmp', ''), 'w')
    meshFile = open(meshFile, 'r')
    cliFile = open(cliFile, 'r')

    fluid_nodes = []
    for line in cliFile:
        if int(line.split()[0]) > 2:
            fluid_nodes.append(int(line.split()[11]))
    cliFile.close()

    found = False
    counter = 0
    for line in meshFile:
        if found:
            if counter == 0:
                num_elements = int(line)
                counter += 1
            else:
                if counter <= num_elements:
                    hack = line.split()
                    hack[2] = '2'
                    if hack[1] == '1':
                        if args_in_list(fluid_nodes, [int(hack[-1]), int(hack[-2])]):
                            add_tag_to_line(hack,'5')
                        else:
                            add_tag_to_line(hack,'1')
                    else:
                        add_tag_to_line(hack,'9')
                    line = ' '.join(hack) + '\n'
                    counter += 1

        if '$Elements' in line:
            found = True

        fixed.write(line)
    fixed.write('\n')
    fixed.close()
    meshFile.close()

if __name__ == '__main__':
    try:
        infile = sys.argv[1]
        clifile = sys.argv[2]
    except IndexError:
        print 'ERROR setting source and/or destination files'
        usage()
        sys.exit(2)

    if infile[-4:]!='.slf' or clifile[-4:]!='.cli':
        print 'Unable to reconcile slf and cli files'
        usage()
        sys.exit(3)

    outfile = infile.split('.')[0]+'_tmp.msh'

    print 'Reading ', infile
    slfData = parserSELAFIN.SELAFIN(infile)

    print 'Extracting nodes, lines, and triangles'
    nodes = get_nodes(slfData)
    lines = get_lines(slfData)
    triangles = slfData.IKLE3

    print 'Filtering internal lines, this may take some time'
    lines = clean_lines(lines, clifile)

    print 'Writing tmp mesh file without physical tags'
    write_tmp_mesh(outfile, nodes, lines, triangles)

    print 'Adding physical tags'
    add_tags_to_gmsh_from_cli(outfile, clifile)

    print 'Converted ', infile, ' to ', outfile
