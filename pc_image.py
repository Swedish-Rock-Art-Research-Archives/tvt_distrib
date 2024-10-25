from PIL import Image
from skimage.morphology import disk
import imageio
import scipy as sp
import multiprocessing as mp
import matplotlib.colorbar as colorbar
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import warnings
import skimage


if __name__ == '__main__':
    mp.freeze_support()

GRAY_COLORMAP = 'gist_yarg'
TOPO_MAP_COLORMAP = 'jet'
DEPTH_MAP_COLORMAP = 'Spectral'
BLENDED_MAP_COLORMAP = 'jet'
MASK_COLORMAP = 'gray'

Image.MAX_IMAGE_PIXELS = None


def clean_mask(mask):

    mask_opened = skimage.morphology.binary_opening(
        mask,
        disk(5)
    )

    mask_cleaned = skimage.morphology.remove_small_objects(
        mask_opened,
        min_size=int(0.00005 * np.sum(mask_opened)),
        connectivity=1
    )

    return mask_cleaned


def enhance_topo_map(topo_map, scan_id):

    q1, q3 = np.percentile(topo_map[~np.isnan(topo_map)].flatten(), [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 7)
    upper_bound = q3 + (iqr * 25)

    enhanced_topo_map_cleaned = topo_map.copy()

    mask = ~np.isnan(enhanced_topo_map_cleaned)
# mask[mask] &= enhanced_topo_map_cleaned[mask] < upper_bound
# mask[mask] &= enhanced_topo_map_cleaned[mask] > lower_bound
    enhanced_topo_map_cleaned[~mask] = np.nan

    mask = ~np.isnan(enhanced_topo_map_cleaned)
    mask_cleaned = skimage.morphology.remove_small_objects(
        mask,
        min_size=int(0.00005 * np.sum(mask)),
        connectivity=1
    )

    enhanced_topo_map_cleaned[~mask_cleaned] = np.nan

    enhanced_topo_map_scaled = enhanced_topo_map_cleaned.copy()

    mask = ~np.isnan(enhanced_topo_map_scaled)
    mask[mask] &= enhanced_topo_map_scaled[mask] >= 0.0
    enhanced_topo_map_scaled[mask] = np.log(
        enhanced_topo_map_scaled[mask] + 1.0)

    mask = ~np.isnan(enhanced_topo_map_scaled)
    mask[mask] &= enhanced_topo_map_scaled[mask] < 0.0
    enhanced_topo_map_scaled[mask] = -1.0 * \
        np.log((-1.0 * enhanced_topo_map_scaled[mask]) + 1.0)

    enhanced_topo_map_equalized = enhanced_topo_map_scaled.copy()

    _min = np.nanmin(enhanced_topo_map_equalized)
    _max = np.nanmax(enhanced_topo_map_equalized)

    nan_mask = np.isnan(enhanced_topo_map_equalized)
    enhanced_topo_map_equalized = (
        enhanced_topo_map_equalized - _min) / (_max - _min)
    enhanced_topo_map_equalized[nan_mask] = 0.0

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        enhanced_topo_map_equalized = skimage.img_as_uint(
            enhanced_topo_map_equalized)
        enhanced_topo_map_equalized = skimage.exposure.equalize_adapthist(
            enhanced_topo_map_equalized,
            kernel_size=1000,
            clip_limit=0.05
        )
        enhanced_topo_map_equalized[nan_mask] = np.nan

    return enhanced_topo_map_equalized


def flatten_normal_map(normal_map):

    sigma = 64.0

    normal_map_mask = np.isnan(normal_map)
    normal_map_smoothed = normal_map.copy()
    normal_map_smoothed[normal_map_mask] = 0.0
    normal_map_smoothed = skimage.filters.gaussian(
        normal_map_smoothed, sigma=sigma, multichannel=True)
    normal_map_smoothed_nans = skimage.filters.gaussian(
        (~normal_map_mask).astype(float), sigma=sigma, multichannel=True)

    normal_map_smoothed = np.true_divide(
        normal_map_smoothed,
        normal_map_smoothed_nans,
        out=np.zeros_like(normal_map_smoothed),
        where=normal_map_smoothed_nans != 0.0
    )

    norms = np.zeros(
        (normal_map_smoothed.shape[0], normal_map_smoothed.shape[1], 3))
    norms[:, :, 0] = np.linalg.norm(normal_map_smoothed, axis=2)
    norms[:, :, 1] = norms[:, :, 0]
    norms[:, :, 2] = norms[:, :, 0]
    normal_map_smoothed = np.true_divide(
        normal_map_smoothed,
        norms,
        out=np.full_like(normal_map_smoothed, np.nan),
        where=norms != 0.0
    )

    flat_normal_map = normal_map + \
        (np.array([0.0, 0.0, 1.0]) - normal_map_smoothed)
    norms = np.zeros((flat_normal_map.shape[0], flat_normal_map.shape[1], 3))
    norms[:, :, 0] = np.linalg.norm(flat_normal_map, axis=2)
    norms[:, :, 1] = norms[:, :, 0]
    norms[:, :, 2] = norms[:, :, 0]
    flat_normal_map = np.true_divide(
        flat_normal_map,
        norms,
        out=np.zeros_like(flat_normal_map),
        where=norms != 0.0
    )

    nm_min = -1.0
    nm_max = 1.0
    normal_map_scaled = (normal_map - nm_min) / (nm_max - nm_min)
    normal_map_smoothed_scaled = (
        normal_map_smoothed - nm_min) / (nm_max - nm_min)
    flat_normal_map_scaled = (flat_normal_map - nm_min) / (nm_max - nm_min)

    return flat_normal_map


def create_topo_map(depth_map, scan_id, sigma):

    depth_map_mask = np.isnan(depth_map)
    depth_map_smoothed = depth_map.copy()
    depth_map_smoothed[depth_map_mask] = 0.0
    depth_map_smoothed = skimage.filters.gaussian(
        depth_map_smoothed, sigma=sigma)
    depth_map_smoothed_nans = skimage.filters.gaussian(
        (~depth_map_mask).astype(float), sigma=sigma)
    depth_map_smoothed = np.true_divide(
        depth_map_smoothed,
        depth_map_smoothed_nans,
        out=np.zeros_like(depth_map_smoothed),
        where=depth_map_smoothed_nans != 0.0
    )
    topo_map = depth_map - depth_map_smoothed

    return topo_map


def generate_images(pcd, scan_id):

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    kd_tree = sp.spatial.cKDTree(np.c_[points[:, 0], points[:, 1]])
    dist, _ = kd_tree.query(
        np.c_[points[:, 0], points[:, 1]], k=range(2, 6))
    dist_nn = np.array([d.mean() for d in dist])
    pc_resolution = dist_nn.mean()

    x_max = points[:, 0].max()
    x_min = points[:, 0].min()
    y_max = points[:, 1].max()
    y_min = points[:, 1].min()

    # offset = 0.01
    # x_offset = offset * np.abs(x_max - x_min)
    # y_offset = offset * np.abs(y_max - y_min)

    x_offset = 0
    y_offset = 0

    grid_x, grid_y = np.mgrid[
        (x_min-x_offset):(x_max+x_offset):(pc_resolution/2.0),
        (y_min-y_offset):(y_max+y_offset):(pc_resolution/2.0)
    ]

    grid_x = np.transpose(grid_x)
    grid_x = np.flip(grid_x, axis=0)
    grid_y = np.transpose(grid_y)
    grid_y = np.flip(grid_y, axis=0)

    grid_x_flattened = grid_x.flatten()
    grid_y_flattened = grid_y.flatten()

    depth_map = sp.interpolate.griddata(
        (
            points[:, 0],
            points[:, 1]
        ),
        points[:, 2],
        (
            grid_x,
            grid_y
        ),
        method='linear',
        rescale=False
    )

    dist, _ = kd_tree.query(np.c_[grid_x_flattened, grid_y_flattened], k=1)
    dist = dist.reshape(grid_x.shape)
    depth_map[dist > pc_resolution + 3.0*dist_nn.std()] = np.nan

    normal_map = sp.interpolate.griddata(
        (
            points[:, 0],
            points[:, 1]
        ),
        normals[:, :],
        (
            grid_x,
            grid_y
        ),
        method='linear',
        rescale=False
    )

    normal_map[np.isnan(depth_map)] = np.nan

    nm_min = -1.0
    nm_max = 1.0
    normal_map_scaled = (normal_map - nm_min) / (nm_max - nm_min)
    nan_mask = np.isnan(normal_map[:, :, 0])
    normal_map_scaled[nan_mask] = (10.0, 15.0, 60.0)

    return depth_map, normal_map, pc_resolution, np.array([grid_x, grid_y])


def rgb_save_images(save_path, scan_id, depth_map, normal_map, topo_maps, enhanced_topo_maps, transparency, img_type,dpi, export_extra):
    rgb_transparency = (0, 0, 0) if transparency else None
    grey_transparency = 0 if transparency else None
    
    if export_extra:

        with warnings.catch_warnings():

            warnings.simplefilter('ignore')

            cmap = plt.colormaps.get_cmap(DEPTH_MAP_COLORMAP)
            cnorm = colors.Normalize(vmin=np.nanmin(
                depth_map), vmax=np.nanmax(depth_map))
            smap = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap)
            nan_mask = np.isnan(depth_map)

            depth_map_img = depth_map.copy()
            depth_map_img[nan_mask] = 0.0
            depth_map_img = skimage.img_as_ubyte(
                smap.to_rgba(depth_map_img)[:, :, :3])
            depth_map_img[nan_mask] = [0, 0, 0]

            if img_type != 'tif':
                imageio.imwrite(f'{save_path}{scan_id}_depth_map.{img_type}', depth_map_img,
                                transparency=rgb_transparency, quality=10, pixelformat='yuvj444p')
            else:
                depth_map_tif = Image.fromarray(depth_map_img)
                depth_map_tif.save(
                    f'{save_path}{scan_id}_depth_map.{img_type}', dpi=(dpi,dpi), quality=100)

            fig = plt.figure(figsize=(8, 1))
            ax = plt.gca()
            cb = colorbar.ColorbarBase(
                ax, cmap=cmap, norm=cnorm, orientation='horizontal')
            cb.set_label('Depth Colorbar')
            plt.tight_layout()
            plt.savefig(save_path + scan_id+'_depth_map_colorbar.png')
            plt.close()

        with warnings.catch_warnings():

            warnings.simplefilter('ignore')

            nm_min = -1.0
            nm_max = 1.0
            nan_mask = np.isnan(normal_map[:, :, 0])

            normal_map_img = (normal_map - nm_min) / (nm_max - nm_min)
            normal_map_img[nan_mask] = [0.0, 0.0, 0.0]
            normal_map_img = skimage.img_as_ubyte(normal_map_img)

            derivative_map_img = normal_map_img.copy()
            derivative_map_img[:, :, 2] = 0

            if img_type != 'tif':
                imageio.imwrite(save_path + scan_id+'_normal_map.'+img_type, normal_map_img,
                                transparency=rgb_transparency, quality=10, pixelformat='yuvj444p')
                imageio.imwrite(save_path + scan_id+'_derivative_map.'+img_type, derivative_map_img,
                                transparency=rgb_transparency, quality=10, pixelformat='yuvj444p')
            else:
                normal_map_tif = Image.fromarray(normal_map_img)
                normal_map_tif.save(
                    save_path+scan_id+'_normal_map.'+img_type, dpi=(dpi,dpi), quality=100)

                derivative_map_tif = Image.fromarray(derivative_map_img)
                derivative_map_tif.save(
                    save_path+scan_id+'_derivative_map.'+img_type, dpi=(dpi,dpi), quality=100)

    with warnings.catch_warnings():

        warnings.simplefilter('ignore')

        for key, topo_map in topo_maps.items():

            cmap = plt.colormaps.get_cmap(TOPO_MAP_COLORMAP)
            cnorm = colors.Normalize(vmin=np.nanmin(
                topo_map), vmax=np.nanmax(topo_map))
            smap = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap)
            nan_mask = np.isnan(topo_map)

            topo_map_img = topo_map.copy()
            topo_map_img[nan_mask] = 0.0
            topo_map_img = skimage.img_as_ubyte(1.0 - cnorm(topo_map_img))
            topo_map_img[nan_mask] = 0

            topo_map_img = topo_map.copy()
            topo_map_img[nan_mask] = 0.0
            topo_map_img = skimage.img_as_ubyte(
                smap.to_rgba(topo_map_img)[:, :, :3])
            topo_map_img[nan_mask] = [0, 0, 0]

            fig = plt.figure(figsize=(8, 1))
            ax = plt.gca()
            cb = colorbar.ColorbarBase(
                ax, cmap=cmap, norm=cnorm, orientation='horizontal')
            cb.set_label('Topo Colorbar')
            plt.tight_layout()
            plt.savefig(save_path + 'topo_maps/'+scan_id +
                        '_topo_map_ ' + key + ' _colorbar.png')
            plt.close()

            if img_type != 'tif':
                imageio.imwrite(save_path + 'topo_maps/'+scan_id+'_topo_map_' + key + '.'+img_type,
                                topo_map_img, transparency=rgb_transparency, quality=10, pixelformat='yuvj444p')

            else:
                topo_map_tif = Image.fromarray(topo_map_img)
                topo_map_tif.save(save_path+'topo_maps/'+scan_id+'_topo_map_' +
                                  key + '.'+img_type, dpi=(dpi,dpi), quality=100)

    with warnings.catch_warnings():

        warnings.simplefilter('ignore')

        for key, enhanced_topo_map in enhanced_topo_maps.items():

            cmap = plt.colormaps.get_cmap(TOPO_MAP_COLORMAP)
            smap = plt.cm.ScalarMappable(norm=None, cmap=cmap)
            nan_mask = np.isnan(enhanced_topo_map)

            enhanced_topo_map_img = enhanced_topo_map.copy()
            enhanced_topo_map_img[nan_mask] = 0.0
            enhanced_topo_map_img = skimage.img_as_ubyte(
                1.0 - enhanced_topo_map_img)
            enhanced_topo_map_img[nan_mask] = 0

            enhanced_topo_map_img = enhanced_topo_map.copy()
            enhanced_topo_map_img[nan_mask] = 0.0
            enhanced_topo_map_img = skimage.img_as_ubyte(
                smap.to_rgba(enhanced_topo_map_img)[:, :, :3])
            enhanced_topo_map_img[nan_mask] = [0, 0, 0]

            fig = plt.figure(figsize=(8, 1))
            ax = plt.gca()
            cb = colorbar.ColorbarBase(
                ax, cmap=cmap, norm=None, orientation='horizontal')
            cb.set_label('Topo Colorbar')
            plt.tight_layout()
            plt.savefig(save_path + 'enhanced_topo_maps/'+scan_id +
                        '_enhanced_topo_map_ ' + key + ' _colorbar.png')
            plt.close()

            if img_type != 'tif':
                imageio.imwrite(save_path + 'enhanced_topo_maps/'+scan_id+'_enhanced_topo_map_' + key + '.'+img_type,
                                enhanced_topo_map_img, transparency=rgb_transparency, quality=10, pixelformat='yuvj444p')
            else:
                enhanced_topo_tif = Image.fromarray(enhanced_topo_map_img)
                enhanced_topo_tif.save(save_path+'enhanced_topo_maps/'+scan_id +
                                       '_enhanced_topo_map_' + key + '.'+img_type, dpi=(dpi,dpi), quality=100)

    with warnings.catch_warnings():

        warnings.simplefilter('ignore')

        for key, enhanced_topo_map in enhanced_topo_maps.items():

            nm_min = -1.0
            nm_max = 1.0
            nan_mask = np.isnan(normal_map[:, :, 0])

            texture_map_img = (normal_map - nm_min) / (nm_max - nm_min)
            texture_map_img = skimage.color.rgb2gray(texture_map_img)

            texture_map_img[nan_mask] = [0.0]
            texture_map_img = skimage.exposure.equalize_adapthist(
                texture_map_img,
                kernel_size=100,
                clip_limit=0.02
            )
            texture_map_img[nan_mask] = [0.0]
            texture_map_img = skimage.img_as_ubyte(texture_map_img)

            if img_type != 'tif':
                imageio.imwrite(save_path + scan_id+'_texture_map.'+img_type, texture_map_img,
                                transparency=grey_transparency, quality=10, pixelformat='yuvj444p')
            else:
                texture_map_tif = Image.fromarray(texture_map_img)
                texture_map_tif.save(
                    save_path + scan_id+'_texture_map.'+img_type, dpi=(dpi,dpi), quality=100)

            cmap = plt.colormaps.get_cmap(BLENDED_MAP_COLORMAP)
            smap = plt.cm.ScalarMappable(norm=None, cmap=cmap)
            nan_mask = np.isnan(enhanced_topo_map)
            texture_map_img[nan_mask] = 0

            blended_map_img = enhanced_topo_map.copy()
            blended_map_img[nan_mask] = 0.0
            blended_map_img = skimage.img_as_ubyte(1.0 - blended_map_img)
            blended_map_img[nan_mask] = 0

            alpha = 0.5

            blended_map_img = enhanced_topo_map.copy()
            blended_map_img[nan_mask] = 0.0
            blended_map_img = skimage.img_as_ubyte(
                smap.to_rgba(blended_map_img)[:, :, :3])
            blended_map_img[nan_mask] = [0, 0, 0]

            alpha = 0.5

            blended_map_img = alpha * blended_map_img + \
                (1 - alpha) * skimage.color.gray2rgb(texture_map_img)
            blended_map_img = blended_map_img.astype(np.uint8)

            if img_type != 'tif':
                imageio.imwrite(save_path + 'blended_maps/'+scan_id+'_blended_map_' + key + '.'+img_type,
                                blended_map_img, transparency=rgb_transparency, quality=10, pixelformat='yuvj444p')
            else:
                blended_map_tif = Image.fromarray(blended_map_img)
                blended_map_tif.save(save_path + 'blended_maps/'+scan_id +
                                     '_blended_map_' + key + '.'+img_type, dpi=(dpi,dpi), quality=100)


def grey_save_images(save_path, scan_id, depth_map, normal_map, topo_maps, enhanced_topo_maps, transparency, img_type, dpi,export_extra):
    rgb_transparency = (0, 0, 0) if transparency else None
    grey_transparency = 0 if transparency else None
    
    if export_extra:
        with warnings.catch_warnings():

            warnings.simplefilter('ignore')

            cmap = plt.colormaps.get_cmap(DEPTH_MAP_COLORMAP)
            cnorm = colors.Normalize(vmin=np.nanmin(
                depth_map), vmax=np.nanmax(depth_map))
            smap = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap)
            nan_mask = np.isnan(depth_map)

            depth_map_img = depth_map.copy()
            depth_map_img[nan_mask] = 0.0
            depth_map_img = skimage.img_as_ubyte(
                smap.to_rgba(depth_map_img)[:, :, :3])
            depth_map_img[nan_mask] = [0, 0, 0]

            if img_type != 'tif':
                imageio.imwrite(save_path + scan_id+'_depth_map.'+img_type, depth_map_img,
                                transparency=rgb_transparency, quality=10, pixelformat='yuvj444p')
            else:
                depth_map_tif = Image.fromarray(depth_map)
                depth_map_tif.save(
                    save_path+scan_id+'_depth_map.'+img_type, dpi=(dpi, dpi), quality=100)

            fig = plt.figure(figsize=(8, 1))
            ax = plt.gca()
            cb = colorbar.ColorbarBase(
                ax, cmap=GRAY_COLORMAP, norm=cnorm, orientation='horizontal')
            cb.set_label('Depth Colorbar')
            plt.tight_layout()
            plt.savefig(save_path + scan_id+'_depth_map_colorbar.png')
            plt.close()

        with warnings.catch_warnings():

            warnings.simplefilter('ignore')

            nm_min = -1.0
            nm_max = 1.0
            nan_mask = np.isnan(normal_map[:, :, 0])

            normal_map_img = (normal_map - nm_min) / (nm_max - nm_min)
            normal_map_img[nan_mask] = [0.0, 0.0, 0.0]
            normal_map_img = skimage.img_as_ubyte(normal_map_img)

            derivative_map_img = normal_map_img.copy()
            derivative_map_img[:, :, 2] = 0

            if img_type != 'tif':
                imageio.imwrite(save_path + scan_id+'_normal_map.'+img_type, normal_map_img,
                                transparency=rgb_transparency, quality=10, pixelformat='yuvj444p')
                imageio.imwrite(save_path + scan_id+'_derivative_map.'+img_type, derivative_map_img,
                                transparency=rgb_transparency, quality=10, pixelformat='yuvj444p')
            else:
                normal_map_tif = Image.fromarray(normal_map_img)
                normal_map_tif.save(
                    save_path+scan_id+'_normal_map.'+img_type, dpi=(dpi, dpi), quality=100)

                derivative_map_tif = Image.fromarray(derivative_map_img)
                derivative_map_tif.save(
                    save_path+scan_id+'_derivative_map.'+img_type, dpi=(dpi, dpi), quality=100)

    with warnings.catch_warnings():

        warnings.simplefilter('ignore')

        for key, topo_map in topo_maps.items():

            cmap = plt.colormaps.get_cmap(TOPO_MAP_COLORMAP)
            cnorm = colors.Normalize(vmin=np.nanmin(
                topo_map), vmax=np.nanmax(topo_map))
            smap = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap)
            nan_mask = np.isnan(topo_map)

            topo_map_img = topo_map.copy()
            topo_map_img[nan_mask] = 0.0
            topo_map_img = skimage.img_as_ubyte(1.0 - cnorm(topo_map_img))
            topo_map_img[nan_mask] = 0

            if img_type != 'tif':
                imageio.imwrite(save_path + 'topo_maps/'+scan_id+'_topo_map_' + key + '_grey.'+img_type,
                                topo_map_img, transparency=grey_transparency, quality=10, pixelformat='yuvj444p')

            else:
                topo_map_tif = Image.fromarray(topo_map_img)
                topo_map_tif.save(save_path + 'topo_maps/'+scan_id+'_topo_map_' +
                                  key + '_grey.'+img_type, dpi=(dpi, dpi), quality=100)

            topo_map_img = topo_map.copy()
            topo_map_img[nan_mask] = 0.0
            topo_map_img = skimage.img_as_ubyte(
                smap.to_rgba(topo_map_img)[:, :, :3])
            topo_map_img[nan_mask] = [0, 0, 0]

    with warnings.catch_warnings():

        warnings.simplefilter('ignore')

        for key, enhanced_topo_map in enhanced_topo_maps.items():

            cmap = plt.colormaps.get_cmap(TOPO_MAP_COLORMAP)
            smap = plt.cm.ScalarMappable(norm=None, cmap=cmap)
            nan_mask = np.isnan(enhanced_topo_map)

            enhanced_topo_map_img = enhanced_topo_map.copy()
            enhanced_topo_map_img[nan_mask] = 0.0
            enhanced_topo_map_img = skimage.img_as_ubyte(
                1.0 - enhanced_topo_map_img)
            enhanced_topo_map_img[nan_mask] = 0

            if img_type != 'tif':
                imageio.imwrite(save_path + 'enhanced_topo_maps/'+scan_id+'_enhanced_topo_map_' + key + '_grey.' +
                                img_type, enhanced_topo_map_img, transparency=grey_transparency, quality=10, pixelformat='yuvj444p')

            else:
                enhanced_topo_tif = Image.fromarray(enhanced_topo_map_img)
                enhanced_topo_tif.save(save_path + 'enhanced_topo_maps/'+scan_id +
                                       '_enhanced_topo_map_' + key + '_grey.'+img_type, dpi=(dpi, dpi), quality=100)

            enhanced_topo_map_img = enhanced_topo_map.copy()
            enhanced_topo_map_img[nan_mask] = 0.0
            enhanced_topo_map_img = skimage.img_as_ubyte(
                smap.to_rgba(enhanced_topo_map_img)[:, :, :3])
            enhanced_topo_map_img[nan_mask] = [0, 0, 0]

    with warnings.catch_warnings():

        warnings.simplefilter('ignore')

        for key, enhanced_topo_map in enhanced_topo_maps.items():

            nm_min = -1.0
            nm_max = 1.0
            nan_mask = np.isnan(normal_map[:, :, 0])

            texture_map_img = (normal_map - nm_min) / (nm_max - nm_min)
            texture_map_img = skimage.color.rgb2gray(texture_map_img)

            texture_map_img[nan_mask] = [0.0]
            texture_map_img = skimage.exposure.equalize_adapthist(
                texture_map_img,
                kernel_size=100,
                clip_limit=0.02
            )
            texture_map_img[nan_mask] = [0.0]
            texture_map_img = skimage.img_as_ubyte(texture_map_img)
            texture_map_img2 = Image.fromarray(
                texture_map_img.astype(np.uint8))

            if img_type != 'tif':
                texture_map_img2.save(
                    save_path + scan_id+'_texture_map.'+img_type, dpi=(dpi, dpi), compress_level=0)
                # imageio.imwrite(save_path + scan_id+'_texture_map.'+img_type, texture_map_img,
                #                 transparency=grey_transparency, quality=10, pixelformat='yuv444p')
            else:
                texture_map_tif = Image.fromarray(texture_map_img)
                texture_map_tif.save(
                    save_path + scan_id+'_texture_map.'+img_type, dpi=(dpi, dpi), quality=100)

            cmap = plt.colormaps.get_cmap(BLENDED_MAP_COLORMAP)
            smap = plt.cm.ScalarMappable(norm=None, cmap=cmap)
            nan_mask = np.isnan(enhanced_topo_map)
            # texture_map_img[nan_mask] = 0

            blended_map_img = enhanced_topo_map.copy()
            blended_map_img[nan_mask] = 0.0
            blended_map_img = skimage.img_as_ubyte(1.0 - blended_map_img)
            blended_map_img[nan_mask] = 0

            alpha = 0.5

            blended_map_img = alpha * blended_map_img + \
                (1 - alpha) * texture_map_img
            blended_map_img = blended_map_img.astype(np.uint8)
            blended_map_img = Image.fromarray(blended_map_img)

            if img_type != 'tif':
                blended_map_img.save(
                    save_path + 'blended_maps/'+scan_id+'_blended_map_' + key + '_grey.'+img_type, dpi=(dpi, dpi), compress_level=0)
                # imageio.imwrite(save_path + 'blended_maps/'+scan_id+'_blended_map_' + key + '_grey.'+img_type,
                #                 blended_map_img, transparency=grey_transparency, quality=10, pixelformat='yuvj444p')
            else:
                blended_map_tif = Image.fromarray(blended_map_img)
                blended_map_tif.save(save_path + 'blended_maps/'+scan_id+'_blended_map_' +
                                     key + '_grey.'+img_type, dpi=(dpi, dpi), quality=100)
