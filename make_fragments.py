# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/make_fragments.py

import numpy as np
import math
import open3d as o3d
import sys

sys.path.append("../utility")
from file import join, make_clean_folder, get_rgbd_file_lists
from opencv import initialize_opencv

sys.path.append(".")
from optimize_posegraph import optimize_posegraph_for_fragment

# check opencv python package
with_opencv = initialize_opencv()
if with_opencv:
    from opencv_pose_estimation import pose_estimation


# step 0
def read_rgbd_image(color_file, depth_file, convert_rgb_to_intensity, config):
    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_trunc=config["max_depth"],
        convert_rgb_to_intensity=convert_rgb_to_intensity)
    return rgbd_image


# step 1
# The function reads a pair of RGBD images and registers the source_rgbd_image to the target_rgbd_image
def register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic, with_opencv, config):
    # read rgb and depth to rgbd
    source_rgbd_image = read_rgbd_image(color_files[s], depth_files[s], True, config)
    target_rgbd_image = read_rgbd_image(color_files[t], depth_files[t], True, config)

    option = o3d.pipelines.odometry.OdometryOption()
    option.max_depth_diff = config["max_depth_diff"]

    if abs(s - t) != 1:  # 如果误差在范围内？
        if with_opencv:
            # pose_estimation() computes OpenCV ORB feature to match sparse features over wide baseline images,
            # then performs 5-point RANSAC to estimate a rough alignment
            # 这个地方，意思是如果误差过大，先算出大概的估计 odo_init
            success_5pt, odo_init = pose_estimation(source_rgbd_image,
                                                    target_rgbd_image,
                                                    intrinsic, False)
            if success_5pt:  # 如果粗配准成功了，则进行细配准
                [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
                    source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    option)
                return [success, trans, info]

        return [False, np.identity(4), np.identity(6)]

    else:
        odo_init = np.identity(4)  # 初始化一个变换 an identity matrix is used as the initialization.
        # compute_rgbd_odometry() is called to align the RGBD images. 返回两帧之间的变换
        [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

        return [success, trans, info]


# Multiway registration
# implement of 《Robust Reconstruction of Indoor Scenes, CVPR, 2015.》
# step 2

# this function builds a pose graph for multiway registration of all RGBD images in this sequence.
# Each graph node represents an RGBD image and its pose which transforms the geometry to the global fragment space.
# For efficiency, only key frames are used.
# 为RGBD序列构建pose graph
def make_posegraph_for_fragment(path_dataset,
                                sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic, with_opencv, config):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    trans_odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(trans_odometry))

    for s in range(sid, eid):
        for t in range(s + 1, eid):
            # odometry
            if t == s + 1:
                print(
                    "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                    % (fragment_id, n_fragments - 1, s, t))
                [success, trans,
                 info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                intrinsic, with_opencv, config)
                trans_odometry = np.dot(trans, trans_odometry)
                trans_odometry_inv = np.linalg.inv(trans_odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        trans_odometry_inv))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(s - sid,
                                                             t - sid,
                                                             trans,
                                                             info,
                                                             uncertain=False))

            # keyframe loop closure
            if s % config['n_keyframes_per_n_frame'] == 0 \
                    and t % config['n_keyframes_per_n_frame'] == 0:
                print("Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                      % (fragment_id, n_fragments - 1, s, t))
                [success, trans, info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                                intrinsic, with_opencv, config)
                if success:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            s - sid, t - sid, trans, info, uncertain=True))
    o3d.io.write_pose_graph(
        join(path_dataset, config["template_fragment_posegraph"] % fragment_id),
        pose_graph)


# Once the poses are estimates,
# RGBD integration is used to reconstruct a colored fragment from each RGBD sequence.
# 估计完位姿，根据pose_graph把碎片化的RGBD融合起来  暂时不看具体细节，从大局掌握pipeline
def integrate_rgb_frames_for_fragment(color_files, depth_files, fragment_id,
                                      n_fragments, pose_graph_name, intrinsic,
                                      config):
    pose_graph = o3d.io.read_pose_graph(pose_graph_name)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for i in range(len(pose_graph.nodes)):
        i_abs = fragment_id * config['n_frames_per_fragment'] + i
        print("Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." %
              (fragment_id, n_fragments - 1, i_abs, i + 1, len(pose_graph.nodes)))
        rgbd = read_rgbd_image(color_files[i_abs], depth_files[i_abs], False,
                               config)
        pose = pose_graph.nodes[i].pose
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def make_pointcloud_for_fragment(path_dataset, color_files, depth_files,
                                 fragment_id, n_fragments, intrinsic, config):
    mesh = integrate_rgb_frames_for_fragment(
        color_files, depth_files, fragment_id, n_fragments,
        join(path_dataset,
             config["template_fragment_posegraph_optimized"] % fragment_id),
        intrinsic, config)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd_name = join(path_dataset,
                    config["template_fragment_pointcloud"] % fragment_id)
    o3d.io.write_point_cloud(pcd_name, pcd, False, True)


def process_single_fragment(fragment_id, color_files, depth_files, n_files, n_fragments, config):
    if config["path_intrinsic"]:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(
            config["path_intrinsic"])
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    sid = fragment_id * config['n_frames_per_fragment']
    eid = min(sid + config['n_frames_per_fragment'], n_files)

    make_posegraph_for_fragment(config["path_dataset"], sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic, with_opencv, config)
    optimize_posegraph_for_fragment(config["path_dataset"], fragment_id, config)
    make_pointcloud_for_fragment(config["path_dataset"], color_files,
                                 depth_files, fragment_id, n_fragments,
                                 intrinsic, config)


# Batch processing
# The main function calls each individual function explained above.
def run(config):
    print("making fragments from RGBD sequence.")
    make_clean_folder(join(config["path_dataset"], config["folder_fragment"]))
    [color_files, depth_files] = get_rgbd_file_lists(config["path_dataset"])
    n_files = len(color_files)
    n_fragments = int(math.ceil(float(n_files) / config['n_frames_per_fragment']))

    if config["python_multi_threading"]:
        from joblib import Parallel, delayed
        import multiprocessing
        import subprocess
        MAX_THREAD = min(multiprocessing.cpu_count(), n_fragments)
        Parallel(n_jobs=MAX_THREAD)(delayed(process_single_fragment)(
            fragment_id, color_files, depth_files, n_files, n_fragments, config)
                                    for fragment_id in range(n_fragments))
    else:
        for fragment_id in range(n_fragments):
            process_single_fragment(fragment_id, color_files, depth_files,
                                    n_files, n_fragments, config)


# if __name__ == '__main__':
#     print('make_fragment...')
#     run()