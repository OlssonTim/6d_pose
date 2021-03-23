import numpy as np

from plyfile import PlyData


class MeshUtils():
    def __init__(self):
        pass

    def read_xyz_from_obj(self, pth):
        xyz_lst = []
        with open(pth, 'r') as f:
            for line in f.readlines():
                if 'v ' not in line or line[0] != 'v':
                    continue
                xyz_str = [
                    item.strip() for item in line.split(' ')
                    if len(item.strip()) > 0 and 'v' not in item
                ]
                xyz = np.array(items[0:3]).astype(np.float)
                xyz_lst.append(xyz)
        return np.array(xyz_lst)

    def load_ply_model(self, model_path, scale2m=1., ret_dict=True):
        ply = PlyData.read(model_path)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        r = data['red']
        g = data['green']
        b = data['blue']
        face_raw = ply.elements[1].data
        face = []
        for item in face_raw:
            face.append(item[0])

        n_face = len(face)
        face = np.array(face).flatten()
        n_pts = len(x)

        xyz = np.stack([x, y, z], axis=-1) / scale2m
        if not ret_dict:
            return n_pts, xyz, r, g, b, n_face, face
        else:
            ret_dict = dict(
                n_pts=n_pts, xyz=xyz, r=r, g=g, b=b, n_face=n_face, face=face
            )
            return ret_dict

    # Read object vertexes from ply file
    def get_p3ds_from_ply(self, ply_pth, scale2m=1.):
        print("loading p3ds from ply:", ply_pth)
        ply = PlyData.read(ply_pth)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        p3ds = np.stack([x, y, z], axis=-1)
        p3ds = p3ds / float(scale2m)
        print("finish loading ply.")
        return p3ds

    # Read object vertexes from text file
    def get_p3ds_from_txt(self, pxyz_pth):
        pointxyz = np.loadtxt(pxyz_pth, dtype=np.float32)
        return pointxyz

    # Compute the 3D bounding box from object vertexes
    def get_3D_bbox(self, pcld, small=False):
        min_x, max_x = pcld[:, 0].min(), pcld[:, 0].max()
        min_y, max_y = pcld[:, 1].min(), pcld[:, 1].max()
        min_z, max_z = pcld[:, 2].min(), pcld[:, 2].max()
        bbox = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ])
        if small:
            center = np.mean(bbox, 0)
            bbox = (bbox - center[None, :]) * 2.0 / 3.0 + center[None, :]
        return bbox

    # Compute the radius of object
    def get_r(self, bbox):
        return np.linalg.norm(bbox[7,:] - bbox[0,:]) / 2.0

    # Compute the center of object
    def get_centers_3d(self, corners_3d):
        centers_3d = (np.max(corners_3d, 0) + np.min(corners_3d, 0)) / 2
        return centers_3d