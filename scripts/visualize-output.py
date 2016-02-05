#! /usr/bin/env python
"""
Lauches ParaView visualization.

User provides ParaView state file and output directory.

Tuomas Karna 2015-12-02
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
    """
    template_header = """<?xml version="1.0"?>\n"""
    template_openblock = """<VTKFile byte_order="LittleEndian" version="0.1" type="Collection">\n<Collection>\n"""
    template_closeblock = """</Collection>\n</VTKFile>\n"""
    template_entry = """<DataSet timestep="{i}" part="0" group="" file="{name}_{i}.{ext}"/>\n"""
    extension = 'pvtu' if usepvtu else 'vtu'

    content = template_header
    content += template_openblock
    for i in timesteps:
        content += template_entry.format(i=i, name=fieldname, ext=extension)
    content += template_closeblock

    filename = os.path.join(outdir, fieldname+'.pvd')
    print 'generating', filename
    f = open(filename, 'w')
    f.write(content)
    f.close()


def process_args(outputdir, state_file, regenerate_pvd=True, timesteps=None, parallel_vtu=True):
    default_out_dir = 'outputs'
    temp_state_file = os.path.join(TMP_DIR, 'tmp.pvsm')
    paraview_bin = 'paraview'
    pv_log_file = os.path.join(TMP_DIR, 'log_pvoutput.txt')
    static_pvd_files = ['bath']  # outputs that are not time dependent
    # regenerate all existing PVD files
    if regenerate_pvd:
        pvd_files = glob.glob(os.path.join(outputdir, '*.pvd'))
        for f in pvd_files:
            path, fname = os.path.split(f)
            fieldName, extension = os.path.splitext(fname)
            if fieldName not in static_pvd_files:
                generate_pvd_file(outputdir, fieldName, timesteps, usepvtu=parallel_vtu)
    # read state file, replace directory with new one
    new_content = ''
    with open(state_file, 'r') as f:
        content = f.read()
        new_content = content.replace(default_out_dir, outputdir)
    with open(temp_state_file, 'w') as f:
        f.write(new_content)
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
                        help='last time step for regenerated PVD file')
    parser.add_argument('-l', '--last-time-step', type=int, default=100,
                        help='last time step for regenerated PVD file')
    parser.add_argument('--default-outputdir', type=str, default='outputs',
                        help='outputdir in *.pvsm file that will be replaced by outputdir')

    args = parser.parse_args()
    timesteps = range(args.first_time_step, args.last_time_step + 1)
    process_args(args.outputdir, args.statefile, regenerate_pvd=args.regenerate_pvd,
                 timesteps=timesteps, parallel_vtu=args.parallel_vtu)
