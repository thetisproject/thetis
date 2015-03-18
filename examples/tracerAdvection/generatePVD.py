"""Generates pvd master file for given variable"""

from optparse import OptionParser
from string import Template
import os

def generatePVDFile(outDir, fieldName, timestamps, usePVTU):
  header_txt = """<?xml version="1.0"?>\n"""
  openblock_txt = """<VTKFile byte_order="LittleEndian" version="0.1" type="Collection">\n<Collection>\n"""
  closeblock_txt = """</Collection>\n</VTKFile>\n"""
  entry_txt = """<DataSet timestep="${i}" part="0" group="" file="${name}_${i}.${ext}"/>\n"""

  template_header = Template(header_txt)
  template_openblock = Template(openblock_txt)
  template_closeblock = Template(closeblock_txt)
  template_entry = Template(entry_txt)

  filename = os.path.join(outDir,fieldName+'.pvd')
  print 'generating', filename
  extension = 'pvtu' if usePVTU else 'vtu'
  f = open(filename,'w')
  f.write(template_header.substitute())
  f.write(template_openblock.substitute())

  for i in timestamps :
    f.write(template_entry.substitute(i=i,name=fieldName,ext=extension))

  f.write(template_closeblock.substitute())
  f.close()


def parseCommandLine() :

  usage = ('Usage: %prog -r refTag -t [csvStationFile] runID1 runID2 ...\n')

  parser = OptionParser()
  parser.add_option('-d', '--outputDir', action='store', type='string',
                      dest='outDir', help='Directory where vtu/pvtu files are stored (default=%default)',
                      default='outputs')
  parser.add_option('-s', '--start', action='store', type='int',
                      dest='startIndex', help='First time stamp (default = %default)',
                      default=0)
  parser.add_option('-e', '--end', action='store', type='int',
                      dest='endIndex', help='Last time stamp')
  parser.add_option('-n', '--fieldName', action='store', type='string',
                      dest='fieldName', help='Name of the field : XXX_0.vtu')
  parser.add_option('-p', '--parallel', action='store_true',
                      dest='usePVTU', help='Set extension to pvtu instead of vtu', default=False)

  (options, args) = parser.parse_args()

  if options.fieldName == None:
    parser.print_help()
    parser.error('fieldName missing')
  if options.endIndex == None:
    parser.print_help()
    parser.error('endIndex missing')

  indices = range(options.startIndex,options.endIndex+1)
  generatePVDFile(options.outDir, options.fieldName, indices,options.usePVTU)
  
if __name__=='__main__' :
  parseCommandLine()



