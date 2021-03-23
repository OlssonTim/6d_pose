import math
import numpy as np
import ctypes as ct
import cv2

from utils import MeshUtils, PoseUtils, ImgPcldUtils

try:
    from neupeak.utils.webcv2 import imshow, waitKey
except Exception:
    from cv2 import imshow, waitKey

SO_P = './utils/rastertriangle_so.so'

RENDERER = np.ctypeslib.load_library(SO_P, '.')

mesh_utils = MeshUtils()
pose_utils = PoseUtils()
img_pcld_utils = ImgPcldUtils()

def gen_one_zbuf_render(meshc, RT):
    h = 480
    w = 640
    K = [700, 0, 320, 0, 700, 240, 0, 0, 1] # Camera intrinsics
    vis = True
    #if args.extractor == 'SIFT':
    #    extractor = cv2.xfeatures2d.SIFT_create()
    #else:  # use orb
    K = np.array(K).reshape(3, 3)
    extractor = cv2.ORB_create()

    h, w #  = args.h, args.w
    if type(K) == list:
        K = np.array(K).reshape(3, 3)
    R, T = RT[:3, :3], RT[:3, 3]

    new_xyz = meshc['xyz'].copy()
    new_xyz = np.dot(new_xyz, R.T) + T
    p2ds = np.dot(new_xyz.copy(), K.T)
    p2ds = p2ds[:, :2] / p2ds[:, 2:]
    p2ds = np.require(p2ds.flatten(), 'float32', 'C')

    zs = np.require(new_xyz[:, 2].copy(), 'float32', 'C')
    zbuf = np.require(np.zeros(h*w), 'float32', 'C')
    rbuf = np.require(np.zeros(h*w), 'int32', 'C')
    gbuf = np.require(np.zeros(h*w), 'int32', 'C')
    bbuf = np.require(np.zeros(h*w), 'int32', 'C')

    RENDERER.rgbzbuffer(
        ct.c_int(h),
        ct.c_int(w),
        p2ds.ctypes.data_as(ct.c_void_p),
        new_xyz.ctypes.data_as(ct.c_void_p),
        zs.ctypes.data_as(ct.c_void_p),
        meshc['r'].ctypes.data_as(ct.c_void_p),
        meshc['g'].ctypes.data_as(ct.c_void_p),
        meshc['b'].ctypes.data_as(ct.c_void_p),
        ct.c_int(meshc['n_face']),
        meshc['face'].ctypes.data_as(ct.c_void_p),
        zbuf.ctypes.data_as(ct.c_void_p),
        rbuf.ctypes.data_as(ct.c_void_p),
        gbuf.ctypes.data_as(ct.c_void_p),
        bbuf.ctypes.data_as(ct.c_void_p),
    )

    zbuf.resize((h, w))
    msk = (zbuf > 1e-8).astype('uint8')
    if len(np.where(msk.flatten() > 0)[0]) < 500:
        return None
    zbuf *= msk.astype(zbuf.dtype)  # * 1000.0

    bbuf.resize((h, w)), rbuf.resize((h, w)), gbuf.resize((h, w))
    bgr = np.concatenate((bbuf[:, :, None], gbuf[:, :, None], rbuf[:, :, None]), axis=2)
    bgr = bgr.astype('uint8')

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    #if True:
        #imshow("bgr", bgr.astype("uint8"))
        #show_zbuf = zbuf.copy()
        #min_d, max_d = show_zbuf[show_zbuf > 0].min(), show_zbuf.max()
        #show_zbuf[show_zbuf > 0] = (show_zbuf[show_zbuf > 0] - min_d) / (max_d - min_d) * 255
        #show_zbuf = show_zbuf.astype(np.uint8)
        #imshow("dpt", show_zbuf)
        #show_msk = (msk / msk.max() * 255).astype("uint8")
        #imshow("msk", show_msk)
        #waitKey(0)

    data = {}
    data['depth'] = zbuf
    data['rgb'] = rgb
    data['mask'] = msk
    data['K'] = K
    data['RT'] = RT
    data['cls_typ'] = "ape"
    data['rnd_typ'] = 'render'

    kps, des = extractor.detectAndCompute(bgr, None)

    kp_xys = np.array([kp.pt for kp in kps]).astype(np.int32)
    kp_idxs = (kp_xys[:, 1], kp_xys[:, 0])

    dpt_xyz = img_pcld_utils.dpt_2_cld(zbuf, 1.0, K)
    kp_x = dpt_xyz[:, :, 0][kp_idxs][..., None]
    kp_y = dpt_xyz[:, :, 1][kp_idxs][..., None]
    kp_z = dpt_xyz[:, :, 2][kp_idxs][..., None]
    kp_xyz = np.concatenate((kp_x, kp_y, kp_z), axis=1)

    # filter by dpt (pcld)
    kp_xyz, msk = img_pcld_utils.filter_pcld(kp_xyz)
    kps = [kp for kp, valid in zip(kps, msk) if valid]  # kps[msk]
    des = des[msk, :]

    # 6D pose of object in cv camer coordinate system
    # transform to object coordinate system
    kp_xyz = (kp_xyz - RT[:3, 3]).dot(RT[:3, :3])
    dpt_xyz = dpt_xyz[dpt_xyz[:, :, 2] > 0, :]
    dpt_pcld = (dpt_xyz.reshape(-1, 3) - RT[:3, 3]).dot(RT[:3, :3])

    data['kp_xyz'] = kp_xyz
    data['dpt_pcld'] = dpt_pcld

    return data

def load_mesh_c(mdl_p, scale2m):
    meshc = mesh_utils.load_ply_model(mdl_p, scale2m=scale2m)
    meshc['face'] = np.require(meshc['face'], 'int32', 'C')
    meshc['r'] = np.require(np.array(meshc['r']), 'float32', 'C')
    meshc['g'] = np.require(np.array(meshc['g']), 'float32', 'C')
    meshc['b'] = np.require(np.array(meshc['b']), 'float32', 'C')

    return meshc

def extract_textured():
    n_latitude = 3 # Number of latitudes on sphere to sample
    n_longitude = 3 # Number of longitudes on sphere to sample
    mesh_path = 'utils/ape.ply' # Test object
    scale2m = 1 # Scale to transform unit of object to be in meter
    mesh = load_mesh_c(mesh_path, scale2m)

    xyzs = mesh['xyz']
    mean = np.mean(xyzs, axis=0)
    obj_pose = np.eye(4)

    bbox = mesh_utils.get_3D_bbox(xyzs)
    r = mesh_utils.get_r(bbox)
    print("r:", r)

    sph_r = r / 0.035 * 0.18
    print("sph:", sph_r)

    positions = pose_utils.CameraPositions(
        n_longitude, n_latitude, sph_r
    )

    cam_poses = [pose_utils.getCameraPose(pos) for pos in positions]

    kp3ds = []

    for cam_pose in cam_poses:
        o2c_pose = pose_utils.get_o2c_pose_cv(cam_pose, obj_pose) # Get object 6d pose in cv camera coordinate system
        data = gen_one_zbuf_render(mesh, o2c_pose)
        kp3ds += list(data['kp_xyz'])

    with open("%s_%s_textured_kp3ds.obj" % ("ape", "orb"), 'w') as of:
        for p3d in kp3ds:
            print('v ', p3d[0], p3d[1], p3d[2], file=of)



if __name__ == '__main__':
    extract_textured()
