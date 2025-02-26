from pc_image import *
from pc_point import *
import open3d as o3d
import pandas as pd
import numpy as np
import traceback
import gc
import psutil
import time
import sys
import os
import tkinter as tk
from sklearn.neighbors import *
from datetime import datetime
import multiprocessing
multiprocessing.freeze_support()


class pointvis:

    PROCESS = psutil.Process(os.getpid())
    GIGA = 10 ** 9

    def log_exception(exception: BaseException, expected: bool = True):
        output = "[{}] {}: {}".format(
            'EXPECTED' if expected else 'UNEXPECTED', type(exception).__name__, exception)

        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback)

        tk.messagebox.showinfo(
            title='Topography Visualisation Toolbox: {}'.format(exc_type), message=output)


    def ms_output(seconds):
        return str(pd.to_timedelta(seconds, unit='s'))

    def create_panel_folder(path, array):
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(path + '/topo_maps')
            os.makedirs(path + '/enhanced_topo_maps')
            os.makedirs(path + '/blended_maps')

            if array:
                os.makedirs(path + '/arrays')

    def start_pv(data_path, save_path, meta_data_path, visualize_steps, downsample_mesh, voxel_multiplier, flip_pcd, update_resolution, scale_multiplier, progress_text, array, files, export_rgb, export_grey, img_type, transparency, clean_noise, auto_rotate, dpi):
        total_time_start = time.time()
        stages = ['Organising input', 'Reading point data', 'Computing normals and processing point cloud', 'Removing outliers',
                  'Transforming point cloud', 'Creating images', 'Creating topo maps', 'Enhancing images', 'Saving images', 'Saving summary', 'Processing finished']

        stage = stages[0]
        progress_text.configure(text=stage)
        progress_text.update()

        nr_scans = 'Nr of Scans to Process: ' + str(len(files))

        meta_data = {
            'id': [],
            'mesh_edge_resolution': [],
            'coord_units_updated': [],
            'mesh_n_vertices': [],
            'pc_n_points': [],
            'pc_cleaned_n_points': [],
            'n_wrong_n_dir': [],
            'rot_z_cc_deg': [],
            'scale': [],
            'scan_width': [],
            'scan_height': [],
            'scan_depth': [],
            'scan_depth_per_m2': [],
            'pc_resolution': [],
            'img_width': [],
            'img_height': [],
            'proc_time': [],
            'rotation': [],
            'cleaning': [],
            'downsampling': [],
            'final_pc_resolution': [],
        }
        o3d.utility.set_verbosity_level(
            o3d.utility.VerbosityLevel.Error)
        scan_counter = 0
        for scan_id, read_path, save_path in files[:]:
            
            try:

                scan_counter += 1
                scan_time_start = time.time()

                # Create Data Folder
                ########################################################

                pointvis.create_panel_folder(
                    os.path.join(str(save_path), scan_id), array)

                meta_data['id'].append(scan_id)
                if auto_rotate:
                    meta_data['rotation'].append('Auto')
                elif flip_pcd:
                    meta_data['rotation'].append('Strict')
                else:
                    meta_data['rotation'].append('None')
                    
                meta_data['cleaning'].append(clean_noise)
                if downsample_mesh:
                    meta_data['downsampling'].append(voxel_multiplier)
                else:
                    meta_data['downsampling'].append(None) 
                meta_data['mesh_n_vertices'].append(None)
                meta_data['mesh_edge_resolution'].append(None)
                meta_data['coord_units_updated'].append(None)
                   

                ########################################################

                # Read Data File
                # Compute Normals
                # Convert to Point Cloud
                ########################################################

                stage = stages[1]
                progress_text.configure(text='{}: {}...'.format(scan_id, stage))
                progress_text.update()

                time_start = time.time()
                stage = stages[2]
                progress_text.configure(text='{}: {}...'.format(scan_id, stage))
                progress_text.update()
                
                if os.path.splitext(read_path)[1] in ['.las','.laz']:
                    input_pcd = conv_las(read_path,downsample=downsample_mesh,
                    voxel_multiplier=voxel_multiplier,visualize = visualize_steps)
                    
                else:
                    input_pcd = conv_pcd(read_path,downsample=downsample_mesh,
                    voxel_multiplier=voxel_multiplier,visualize = visualize_steps)
                    
                stage = stages[3]
                progress_text.configure(text='{}: {}...'.format(scan_id, stage))
                progress_text.update()  
                  
                pcd, pcd_resolution = process_point_cloud(pcd = input_pcd,
                                                          update_resolution = update_resolution, scale_multiplier = scale_multiplier, flip_pcd = flip_pcd, visualize=visualize_steps)


                

                meta_data['pc_resolution'].append(pcd_resolution)

                meta_data['pc_n_points'].append(len(pcd.points))

                if array:
                    o3d.io.write_point_cloud(os.path.join(
                        save_path, scan_id) + '/' + 'original.pcd', pcd)

                time_end = time.time()

                ########################################################

                # Outlier Removal
                ########################################################
                if clean_noise:
                    stage = stages[4]
                    progress_text.configure(text='{}: {}...'.format(scan_id, stage))
                    progress_text.update()
                    time_start = time.time()

                    pcd = noise_removal(
                        pcd,
                        pcd_resolution,
                        visualize=visualize_steps
                    )

                    meta_data['pc_cleaned_n_points'].append(len(pcd.points))

                    if array:
                        o3d.io.write_point_cloud(os.path.join(
                            save_path, scan_id) + '/' + 'cleaned.pcd', pcd)

                    time_end = time.time()
                else:
                    meta_data['pc_cleaned_n_points'].append('Outlier removal skipped')

                ########################################################

                # Transform Point Cloud through PCA
                ########################################################
                if auto_rotate:
                    stage = stages[5]
                    progress_text.configure(text='{}: {}...'.format(scan_id, stage))
                    progress_text.update()
                    time_start = time.time()

                    pcd, scan_stats, components, mean, rotation = transform_point_cloud(
                        pcd,
                        visualize=visualize_steps
                    )

                    meta_data['n_wrong_n_dir'].append(scan_stats['n_wrong_n_dir'])
                    meta_data['rot_z_cc_deg'].append(scan_stats['rot_z_cc_deg'])
                    meta_data['scan_width'].append(scan_stats['width'])
                    meta_data['scan_height'].append(scan_stats['height'])
                    meta_data['scan_depth'].append(scan_stats['depth'])
                    meta_data['scan_depth_per_m2'].append(
                        scan_stats['depth'] / (scan_stats['height'] * scan_stats['width']))

                    meta_data['scale'].append(scale_multiplier)

                    if array:
                        o3d.io.write_point_cloud(os.path.join(
                            save_path, scan_id) + '/' + 'transformed.pcd', pcd)
                        np.savez_compressed(
                            os.path.join(save_path, scan_id) +
                            '/arrays/transform_meta.npz',
                            components=components,
                            mean=mean,
                            rotation=rotation
                        )

                    time_end = time.time()
                    
                else:
                    meta_data['n_wrong_n_dir'].append('Auto-rotation skipped')
                    meta_data['rot_z_cc_deg'].append('Auto-rotation skipped')
                    meta_data['scan_width'].append('Auto-rotation skipped')
                    meta_data['scan_height'].append('Auto-rotation skipped')
                    meta_data['scan_depth'].append('Auto-rotation skipped')
                    meta_data['scan_depth_per_m2'].append(
                        'Auto-rotation skipped')
                    meta_data['scale'].append('Auto-rotation skipped')

                ########################################################

                # Create Images from Point Cloud
                ########################################################

                stage = stages[6]
                progress_text.configure(text='{}: {}...'.format(scan_id, stage))
                progress_text.update()
                time_start = time.time()

                depth_map, normal_map, final_pc_resolution, pix_to_coords_map = generate_images(
                    pcd, scan_id
                )

                meta_data['final_pc_resolution'].append(final_pc_resolution)
                meta_data['img_width'].append(depth_map.shape[1])
                meta_data['img_height'].append(depth_map.shape[0])

                if array:
                    np.savez_compressed(os.path.join(
                        save_path, scan_id) + '/arrays/pix_to_coords_map.npz', data=pix_to_coords_map)
                    np.savez_compressed(os.path.join(
                        save_path, scan_id) + '/arrays/depth_map.npz', data=depth_map)
                    np.savez_compressed(os.path.join(
                        save_path, scan_id) + '/arrays/normal_map.npz', data=normal_map)

                time_end = time.time()

                ########################################################

                # Create Topography Map
                ########################################################

                stage = stages[7]
                progress_text.configure(text='{}: {}...'.format(scan_id, stage))
                progress_text.update()
                time_start = time.time()

                topo_map_texture_level = create_topo_map(
                    depth_map, scan_id,
                    sigma=(4.0 / final_pc_resolution)
                )

                topo_map_object_level = create_topo_map(
                    depth_map, scan_id,
                    sigma=(32.0 / final_pc_resolution)
                )

                if array:
                    np.savez_compressed(os.path.join(
                        save_path, scan_id) + '/arrays/topo_map_texture_level.npz', data=topo_map_texture_level)
                    np.savez_compressed(os.path.join(
                        save_path, scan_id) + '/arrays/topo_map_object_level.npz', data=topo_map_object_level)

                time_end = time.time()

                ########################################################

                # Enhance Topo Map
                ########################################################

                stage = stages[8]
                progress_text.configure(text='{}: {}...'.format(scan_id, stage))
                progress_text.update()
                time_start = time.time()

                enhanced_topo_map_texture_level = enhance_topo_map(
                    topo_map_texture_level,
                    scan_id
                )

                mask = clean_mask(
                    ~np.isnan(enhanced_topo_map_texture_level)
                )

                enhanced_topo_map_object_level = enhance_topo_map(
                    topo_map_object_level,
                    scan_id
                )

                enhanced_topo_map_texture_level[~mask] = np.nan
                enhanced_topo_map_object_level[~mask] = np.nan

                if array:
                    np.savez_compressed(os.path.join(
                        save_path, scan_id) + '/arrays/enhanced_topo_map_texture_level.npz', data=enhanced_topo_map_texture_level)
                    np.savez_compressed(os.path.join(
                        save_path, scan_id) + '/arrays/enhanced_topo_map_object_level.npz', data=enhanced_topo_map_object_level)

                time_end = time.time()

                ########################################################

                # Save Images.
                ########################################################

                stage = stages[9]
                progress_text.configure(text='{}: {}...'.format(scan_id, stage))
                progress_text.update()
                time_start = time.time()
                
                if export_rgb and export_grey:
                    extra_rgb = True
                    extra_grey = False
                elif export_rgb and not export_grey:
                    extra_rgb = True
                    extra_grey = False
                elif export_grey and not export_rgb:
                    extra_rgb = False
                    extra_grey = True
                else:
                    extra_rgb = True
                    extra_grey = True

                if export_rgb:
                    rgb_save_images(
                        os.path.join(save_path, scan_id) + '/',
                        scan_id,
                        depth_map,
                        normal_map,
                        {
                            'texture_level': topo_map_texture_level,
                            'object_level': topo_map_object_level
                        },
                        {
                            'texture_level': enhanced_topo_map_texture_level,
                            'object_level': enhanced_topo_map_object_level
                        },
                        transparency, img_type, dpi, extra_rgb
                    )
                if export_grey:
                    grey_save_images(
                        os.path.join(save_path, scan_id) + '/',
                        scan_id,
                        depth_map,
                        normal_map,
                        {
                            'texture_level': topo_map_texture_level,
                            'object_level': topo_map_object_level
                        },
                        {
                            'texture_level': enhanced_topo_map_texture_level,
                            'object_level': enhanced_topo_map_object_level
                        },
                        transparency, img_type, dpi, extra_grey
                    )

                time_end = time.time()

                ########################################################

                # return scan_counter, scan_id#, topo_map_texture_level, topo_map_object_level, enhanced_topo_map_texture_level, enhanced_topo_map_object_level

                del depth_map
                del normal_map
                del pix_to_coords_map
                del topo_map_texture_level
                del topo_map_object_level
                del enhanced_topo_map_texture_level
                del enhanced_topo_map_object_level
                gc.collect()

                scan_time_end = time.time()
                meta_data['proc_time'].append(pd.to_timedelta(
                    scan_time_end - scan_time_start, unit='s'))
                for key,val in meta_data.items():
                    if val==None:
                        meta_data[key].append('ERROR')
                
            except:
                with open(os.path.join(save_path, f"error_{datetime.now().strftime('%y%m%d%H%M%S')}.txt"), 'w') as f:
                    f.write(f"{traceback.format_exc()}\n") if traceback.format_exc() else f.write(f"Error: {sys.exc_info()}\n")
                if scan_id not in meta_data['id']:
                    meta_data['id'].append(scan_id) 
                for key,val in meta_data.items():
                    if val==None:
                        meta_data[key].append('ERROR')
                

        stage = stages[10]
        progress_text.configure(text='{}: {}...'.format(scan_id, stage))
        progress_text.update()
        df_meta_data = pd.DataFrame(meta_data)
        df_meta_data.to_csv(meta_data_path, index=False)

        total_time_end = time.time()