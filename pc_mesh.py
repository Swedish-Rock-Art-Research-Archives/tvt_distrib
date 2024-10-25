import itertools

import numpy as np
import open3d as o3d
import multiprocessing as mp

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import DBSCAN, OPTICS


if __name__ == '__main__':
    mp.freeze_support()


def visualize_point_clouds(pcd_list, step):
    o3d.visualization.draw_geometries(pcd_list, window_name=f'{step} - TVT', width=960, height=540)


def read_mesh(path, down_sample, voxel_multiplier, fill, rdepth, visualize=False):

    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()

    vertices = np.asarray(mesh.vertices)
    random_idx = np.random.choice(len(mesh.triangles), 100000000)
    random_triangles = np.asarray(mesh.triangles)[random_idx]
    random_edges = vertices[random_triangles[:, 1]] - \
        vertices[random_triangles[:, 0]]
    mesh_edge_resolution = np.average(np.linalg.norm(random_edges, axis=1))

    if down_sample:

        '''
        mesh = o3d.simplify_quadric_decimation(
                mesh,
                target_number_of_triangles=100#int(len(mesh.vertices)/2.0)
        )
        '''

        mesh = mesh.simplify_vertex_clustering(
            voxel_size=voxel_multiplier * mesh_edge_resolution,
            contraction=o3d.geometry.SimplificationContraction.Quadric
        )

        vertices = np.asarray(mesh.vertices)
        random_idx = np.random.choice(len(mesh.triangles), 100000)
        random_triangles = np.asarray(mesh.triangles)[random_idx]

        random_edges = vertices[random_triangles[:, 1]
                                ] - vertices[random_triangles[:, 0]]
        mesh_edge_resolution = np.average(np.linalg.norm(random_edges, axis=1))

    if fill:
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.normals = mesh.vertex_normals
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=rdepth)
        # mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        # mesh = mesh.fill_holes(max_hole)
        # mesh = o3d.t.geometry.TriangleMesh.to_legacy(mesh)
        vertices = np.asarray(mesh.vertices)
        random_idx = np.random.choice(len(mesh.triangles), 100000)
        random_triangles = np.asarray(mesh.triangles)[random_idx]

        random_edges = vertices[random_triangles[:, 1]
                                ] - vertices[random_triangles[:, 0]]
        mesh_edge_resolution = np.average(np.linalg.norm(random_edges, axis=1))

    if visualize:
        visualize_point_clouds([mesh],'Mesh')

    return mesh, mesh_edge_resolution


def create_point_cloud(mesh, resolution, update_resolution, scale_multiplier, flip_mesh, visualize=False):
    if flip_mesh:
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.normals = mesh.vertex_normals
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0],
                      [0, 0, -1, 0], [0, 0, 0, 1]])

        # pcd.paint_uniform_color([0.7, 0.7, 0.7])  test delete
        pcd.colors = mesh.vertex_colors  # testing effect on SfM data
        pcd.remove_non_finite_points()  # test to handle NoData error

        # Assume wrong units in coord system if resolution is too low.
        if update_resolution:

            pcd.points = o3d.utility.Vector3dVector(
                scale_multiplier * np.asarray(pcd.points))
            resolution = scale_multiplier * resolution

        if visualize:
            visualize_point_clouds([
            ], 'Flipped Point Cloud')

        return pcd, resolution

    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.normals = mesh.vertex_normals

        # pcd.paint_uniform_color([0.7, 0.7, 0.7])  test delete
        pcd.colors = mesh.vertex_colors  # testing effect on SfM data

        pcd.remove_non_finite_points()  # test to handle NoData error

        # Assume wrong units in coord system if resolution is too low.
        if update_resolution:

            pcd.points = o3d.utility.Vector3dVector(
                scale_multiplier * np.asarray(pcd.points))
            resolution = scale_multiplier * resolution

        if visualize:
            visualize_point_clouds([
                pcd
            ], 'Scaled Point Cloud')

        return pcd, resolution


def check_noise_init(comp, grid_x, grid_y, resolution):

    global g_comp, g_grid_x, g_grid_y, g_res
    g_comp = comp
    g_grid_x = grid_x
    g_grid_y = grid_y
    g_res = resolution


def check_noise_worker(coords):

    mask = (
        (g_comp[:, 0] >= g_grid_x[coords[0]]) &
        (g_comp[:, 0] <= g_grid_x[coords[0]+1]) &
        (g_comp[:, 1] >= g_grid_y[coords[1]]) &
        (g_comp[:, 1] <= g_grid_y[coords[1]+1])
    )

    mask_idx = np.argwhere(mask == True)

    if len(g_comp[mask, 2]) > 2:

        noise_cluster_ratio = 0.2

        model = DBSCAN(
            eps=3.0*g_res,
            min_samples=10,
            metric='euclidean',
            algorithm='kd_tree',
            leaf_size=30,
            n_jobs=1
        )
        model.fit(g_comp[mask, :])

        labels, counts = np.unique(model.labels_, return_counts=True)
        noise_idx = np.array([np.argwhere(model.labels_ == l).ravel() for l, c in zip(
            labels, counts) if (c/len(model.labels_)) < noise_cluster_ratio], dtype=object)
        noise_idx = list(itertools.chain.from_iterable(noise_idx))
        noise_idx.sort()

        return mask_idx[noise_idx].ravel().tolist()

        '''
		pca = IncrementalPCA(n_components=3, batch_size=1000)
		pca.fit(g_comp[mask, :])
		comp_local = pca.transform(g_comp[mask, :])
		diff = comp_local[:, 2]

		gap_threshold = 3.0 # Look for gaps > 3.0 mm
		sorted_idx = np.argsort(diff)
		d_diff = np.diff(diff[sorted_idx])
		d_indices = np.arange(d_diff.shape[0])
		d_splitted = np.split(d_indices, np.argwhere(d_diff > gap_threshold).ravel() + 1)

		if len(d_splitted) > 1:
			
			noise_idx = [d_sublist.tolist() for d_sublist in d_splitted if (d_sublist.shape[0]/d_indices.shape[0]) < 0.1] # Keep groups > 10%
			
			if len(noise_idx) > 0:
				noise_idx = list(itertools.chain.from_iterable(noise_idx))
				return mask_idx[sorted_idx[noise_idx]].ravel().tolist()
			else:
				return []
		
		else:
			return []
		'''
    else:
        return []


def noise_removal(pcd, resolution, visualize=False):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)
        outlier_cloud.paint_uniform_color([0.31, 0.14, 0.65])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        
        visualize_point_clouds([inlier_cloud,outlier_cloud], 'Outliers')

    pcd_inlier, ind = pcd.remove_radius_outlier(nb_points=15,
                                                radius=3.0 * resolution)

    if visualize:
        display_inlier_outlier(pcd, ind)

    return pcd_inlier


def rotate(x_arr, y_arr, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """

    ox = 0.0
    oy = 0.0

    x_rot = ox + np.cos(angle) * (x_arr - ox) - np.sin(angle) * (y_arr - oy)
    y_rot = oy + np.sin(angle) * (x_arr - ox) + np.cos(angle) * (y_arr - oy)

    return x_rot, y_rot


def transform_point_cloud(pcd, visualize=False):

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    pca = IncrementalPCA(n_components=3, batch_size=1000)
    pca.fit(points)
    components = pca.components_
    mean = pca.mean_
    normals_mean = np.dot(normals.mean(axis=0), components.T)

    if np.dot(components[2], np.cross(components[0], components[1])) < 0.0:
        components[2] *= -1.0

    if np.dot(components[2], normals.mean(axis=0)) < 0.0:
        components[0] *= -1.0
        components[2] *= -1.0

    points_transformed = np.dot(points - mean, components.T)
    normals_transformed = np.dot(normals, components.T)

    degrees = []
    ratios = []
    for d in range(0, 180, 1):
        x, y = rotate(
            points_transformed[:, 0],
            points_transformed[:, 1],
            np.radians(d)
        )

        x_diff = x.max() - x.min()
        y_diff = y.max() - y.min()
        degrees.append(d)
        ratios.append(x_diff/y_diff)

    idx = np.argsort(ratios)
    rot_z_cc_deg = degrees[idx[-1]]

    points_transformed[:, 0], points_transformed[:, 1] = rotate(
        points_transformed[:, 0],
        points_transformed[:, 1],
        np.radians(rot_z_cc_deg)
    )

    normals_transformed[:, 0], normals_transformed[:, 1] = rotate(
        normals_transformed[:, 0],
        normals_transformed[:, 1],
        np.radians(rot_z_cc_deg)
    )

    wrong_direction_mask = normals_transformed[:, 2] < 0.0
    normals_transformed[wrong_direction_mask, :] = - \
        1.0 * normals_transformed[wrong_direction_mask, :]

    scan_stats = {
        'width': (points_transformed[:, 0].max() - points_transformed[:, 0].min()),
        'height': (points_transformed[:, 1].max() - points_transformed[:, 1].min()),
        'depth': (points_transformed[:, 2].max() - points_transformed[:, 2].min()),
        'rot_z_cc_deg': rot_z_cc_deg,
        'n_wrong_n_dir': np.sum(wrong_direction_mask)
    }


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_transformed)
    pcd.normals = o3d.utility.Vector3dVector(normals_transformed)

    if visualize:

        visualize_point_clouds([pcd], 'Final Point Cloud')

    return pcd, scan_stats, components, mean, rot_z_cc_deg
