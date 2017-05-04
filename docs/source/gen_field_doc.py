# coding=utf-8

"""
Generates rst file 'fields_list.rst' that documents all fields defined in
field_defs.py
"""

from thetis.field_defs import field_metadata

field_names = sorted(field_metadata.keys())


def gen_entry(name, metadata):
    """Generate a doctsring for field 'name' unsing its metadata dict"""
    unit = metadata['unit'] if metadata['unit'] else '-'
    docstr = """- **{:}**: {:} [{:}]\n
    - output file: {:}.pvd\n""".format(name, metadata['name'], unit, metadata['filename'])
    return docstr

with open('field_list.rst', 'w') as outfile:

    for name in field_names:
        docstr = gen_entry(name, field_metadata[name])
        outfile.write(docstr)

