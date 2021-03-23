import numpy as np

from utils import MeshUtils

mesh_utils = MeshUtils()

def load_mesh_c(mdl_p, scale2m):
    meshc = mesh_utils.load_ply_model(mdl_p, scale2m=scale2m)
    meshc['face'] = np.require(meshc['face'], 'int32', 'C')
    meshc['r'] = np.require(np.array(meshc['r']), 'float32', 'C')
    meshc['g'] = np.require(np.array(meshc['g']), 'float32', 'C')
    meshc['b'] = np.require(np.array(meshc['b']), 'float32', 'C')

    return meshc

def extract_textured():
    mesh_path = 'ape.ply' # Test object
    scale2m = 1 # Scale to transform unit of object to be in meter
    mesh = load_mesh_c(mesh_path, scale2m)


if __name__ == '__main__':
    extract_textured()
