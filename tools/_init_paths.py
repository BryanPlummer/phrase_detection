import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

# add each dataset's code too
flickr_path = osp.join(this_dir, '..', 'data', 'flickr')
add_path(flickr_path)

referit_path = osp.join(this_dir, '..', 'data', 'referit')
add_path(referit_path)

# add external codebases
cite_path = osp.join(this_dir, '..', 'external')
add_path(cite_path)
