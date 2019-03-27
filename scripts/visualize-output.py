#! /usr/bin/env python
"""
Launches a ParaView visualization of a simulation.

User provides saved ParaView state file and output directory.

Usage:

.. code-block:: bash

    visualize-output outputs my_visu_state.pvsm

Opens paraview visualization for state ``my_visu_state.pvsm`` where all pvd
files are read from ``outputs`` directory.

.. code-block:: bash

    visualize-output -r outputs my_visu_state.pvsm

As above but first regenerates all ``*.pvd`` files that contain ``*.vtu`` files
for time indices 0..100. Useful in cases where a shorter pvd file has been
created by another simulation run.

.. code-block:: bash

    visualize-output -r -f 20 -l 200 outputs my_visu_state.pvsm

As above but generates ``*.pvd`` files for time indices 20..200.

.. code-block:: bash

    visualize-output -r outputs my_visu_state.pvsm

As above but first generates all ``*.pvd`` for a parallel run, i.e. it lists
``*.pvtu`` files instead of ``*.vtu`` files.
"""
import argparse
import glob
import os
import subprocess
import tempfile

TMP_DIR = tempfile.gettempdir()


def generate_pvd_file(outdir, fieldname, timesteps, usepvtu=False):
    """
    Generates ParaView PVD XML file fieldName.pvd that contains vtu or ptvu files for
    the given time steps range.

    :arg str outdir: directory where pvd files are stored
    :arg str fieldname: name of the field that appears in vtu/pvtu file names
    :arg timesteps: list of time indices of vtu files to include in the pvd file
    :type timesteps: list of int
    """

    template_header = """<?xml version="1.0" ?>\n"""
    template_openblock = """<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n"""
    template_closeblock = """</Collection>\n</VTKFile>\n"""
    template_entry = """<DataSet timestep="{i}" file="{name}_{i}.{ext}" />"""
    extension = 'pvtu' if usepvtu else 'vtu'

    content = template_header
    content += template_openblock
    for i in timesteps:
        content += template_entry.format(i=i, name=fieldname, ext=extension)
    content += template_closeblock

    filename = os.path.join(outdir, fieldname+'.pvd')
    print('generating {:}'.format(filename))
    f = open(filename, 'w')
    f.write(content)
    f.close()


def replace_path_in_xml(filename, outputfile, new_path):
    """
    Replaces all paths in paraview xml file PVDReader entries.

    :arg str filename: XML file to process
    :arg str outputfile: file where updated XML file is saved
    :arg new_path: a new path for all pvd files

    All PVDReader entries of the form
    <Proxy group="sources" type="PVDReader" ...>
      <Property name="FileName" ...>
        <Element value="some/path/to/a_file.pvd" .../>
        ...
      </Property>
      ...
    </Proxy>

    will be reaplaced by
    <Proxy group="sources" type="PVDReader" ...>
      <Property name="FileName" ...>
        <Element value="new_path/a_file.pvd" .../>
        ...
      </Property>
      ...
    </Proxy>

    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    root = tree.getroot()
    readers = root[0].findall("Proxy[@type='PVDReader']")
    for reader in readers:
        fnameprop = reader.findall("Property[@name='FileName']/Element")[0]
        old_fname = fnameprop.attrib['value']
        path, file = os.path.split(old_fname)
        field, ext = os.path.splitext(file)
        new_fname = os.path.join(new_path, field, file)
        fnameprop.attrib['value'] = new_fname
    tree.write(outputfile)


def process_args(outputdir, state_file, regenerate_pvd=True, timesteps=None,
                 parallel_vtu=True):
    """
    Processes command line arguments
    """
    temp_state_file = os.path.join(TMP_DIR, 'tmp.pvsm')
    paraview_bin = 'paraview'
    pv_log_file = os.path.join(TMP_DIR, 'log_pvoutput.txt')
    static_pvd_files = ['bath']  # outputs that are not time dependent
    # regenerate all existing PVD files
    if regenerate_pvd:
        pvd_files = glob.glob(os.path.join(outputdir, '*/*.pvd'))
        for f in pvd_files:
            path, fname = os.path.split(f)
            fieldName, extension = os.path.splitext(fname)
            if fieldName not in static_pvd_files:
                generate_pvd_file(path, fieldName, timesteps, usepvtu=parallel_vtu)
    # read state file, replace directory with new one
    replace_path_in_xml(state_file, temp_state_file, outputdir)
    # lauch paraview with new independent thread
    log_file = open(pv_log_file, 'w')
    cmd = ' '.join([paraview_bin, '--state={:}'.format(temp_state_file), '>', pv_log_file])
    subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch ParaView visualization',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('outputdir', type=str,
                        help='Directory where .pvd files are stored')
    parser.add_argument('statefile', type=str,
                        help='ParaView *.pvsm state file')
    parser.add_argument('-r', action='store_true', dest='regenerate_pvd',
                        help='regenerate PVD files')
    parser.add_argument('-p', action='store_true', dest='parallel_vtu',
                        help='regenerate PVD files for parallel outputs')
    parser.add_argument('-f', '--first-time-step', type=int, default=0,
                        help='first time step to be included in regenerated PVD file')
    parser.add_argument('-l', '--last-time-step', type=int, default=100,
                        help='last time step to be included in regenerated PVD file')

    args = parser.parse_args()
    timesteps = range(args.first_time_step, args.last_time_step + 1)
    process_args(args.outputdir, args.statefile, regenerate_pvd=args.regenerate_pvd,
                 timesteps=timesteps, parallel_vtu=args.parallel_vtu)
