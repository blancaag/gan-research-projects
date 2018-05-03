#!/usr/bin/env python
'''
Python module for the Surrey Face Model (SFM).

This module reads the face models from .scm format files.

To load the model:
    import sfm
    shape_only = False  # or True
    model = sfm.load('path/to/ShpVtxModelBin.scm', shape_only)
    # model is a MorphModel instance.

For a simple demo run:
    python sfm.py path/to/sfm_shape_3448.scm

'''
import struct


class PCAModel():
  def __init__(self, mean, basis, variance):
    self.mean     = mean      # list of doubles [x1,y1,z1,x2,y2,z2,...]
    self.basis    = basis     # list of list of doubles, i.e. a list of eigenvectors
    self.variance = variance  # list of doubles, i.e. a list of eigenvalues
    self.numcomponents = len(basis)
    assert(len(variance) == len(basis))
    assert(len(mean) == len(basis[0]))

class MorphModel():
  def __init__(self, faces, shape, texture=None):
    self.faces   = faces      # list of tuples (each tuple is a triplet of vertex indices)
    self.shape   = shape      # PCAModel instance
    self.texture = texture    # PCAModel instance or None (for shape-only models)
    self.numvertices = max(vx for tri in faces for vx in tri) + 1
    self.numfaces    = len(faces)
    assert(3*self.numvertices == len(shape.mean))
    if texture:
      assert(3*self.numvertices == len(texture.mean))


def load(filepath, shape_only=False):
  def readpca(f, nv):
    nc,nd  = struct.unpack('<II', f.read(8))
    assert(nd == 3*nv)
    funcs  = [[struct.unpack('<d', f.read(8))[0] for d in range(nd)]
              for c in range(nc)]
    (nm,)  = struct.unpack('<I', f.read(4))
    assert(nm == 3*nv)
    # fyi: elements are ordered x1,y1,z1,x2,y2,z2,...xnv,ynv,znv.
    vmean  = [struct.unpack('<d', f.read(8))[0] for i in range(nm)]
    (ne,)  = struct.unpack('<I', f.read(4))
    values = [struct.unpack('<d', f.read(8))[0] for i in range(ne)]
    return PCAModel(vmean, funcs, values)
  
  with open(filepath, 'rb') as f:
    nv,nt = struct.unpack('<II', f.read(8))
    # fyi: faces is a list of tuples (each tuple contains 3 vertex indices).
    faces = [struct.unpack('<III', f.read(12)) for i in range(nt)]
    shape = readpca(f, nv)
    try:
      textr = None if shape_only else readpca(f, nv)
    except Exception:
      print('No texture data. Returning shape-only model.')
      textr = None
  
  return MorphModel(faces, shape, textr)


if __name__ == '__main__':
  import sys
  if len(sys.argv) < 2:
    print(__doc__)
    sys.exit()
  
  filename = sys.argv[1]
  print(filename)
  
  # Assume shape + texture, but will fall back if the model is shape-only.
  model = load(filename)
  
  print('Faces:    {:6d}'.format(model.numfaces))
  print('Vertices: {:6d}'.format(model.numvertices))
  print('Shape:    {:6d} components'.format(model.shape.numcomponents))
  print('Texture:  {:6d} components'.format(model.texture.numcomponents if model.texture else 0))


