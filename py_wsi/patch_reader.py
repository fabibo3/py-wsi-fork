'''

These functions acess the .svs files via the openslide-python API and perform patch
sampling as is typically needed for deep learning.

Author: @ysbecca, Fabian Bongratz
'''
import numpy as np
from PIL import Image, ImageDraw
from openslide import open_slide  
from openslide.deepzoom import DeepZoomGenerator
from glob import glob
from xml.dom import minidom
from shapely.geometry import Polygon, Point

from .store import *

def check_label_exists(label, label_map):
    ''' Checking if a label is a valid label. 
    '''
    if label in label_map:
        return True
    else:
        print("py_wsi error: provided label " + str(label) + " not present in label map.")
        print("Setting label as -1 for UNRECOGNISED LABEL.")
        print(label_map)
        return False
def generate_segmentation_patch(seg_map_full, point, patch_size):
    """
    Generates a segmentation map for a certain patch given the level-size
    segmentation map
    - seg_map_full: the segmentation map for the entire level as Image 
    - point:        the coordinates of the tile within the level
    - patch_size:   size of the patches
    """
    # Choose segmentation patch
    box = ([point[0], point[1], point[0]+patch_size, point[1]+patch_size])
    seg_patch = seg_map_full.crop(box)
    #seg_patch.show()
    return seg_patch


def generate_label(regions, region_labels, point, label_map):
    ''' Generates a label given an array of regions.
        - regions               array of vertices
        - region_labels         corresponding labels for the regions
        - point                 x, y tuple
        - label_map             the label dictionary mapping string labels to integer labels
    '''
    for i in range(len(region_labels)):
        poly = Polygon(regions[i])
        if poly.contains(Point(point[0], point[1])):
            if check_label_exists(region_labels[i], label_map):
                return label_map[region_labels[i]]
            else:
                return -1
    # By default, we set to "Normal" if it exists in the label map.
    if check_label_exists('Normal', label_map):
        return label_map['Normal']
    else:
        return -1

def transform_regions(regions, image_size, total_image_size):
    """
    Transform the regions in regions based on the difference in image_size and
    total_image_size
    """
    factor_x = float(image_size[0])/total_image_size[0]
    factor_y = float(image_size[1])/total_image_size[1]
    factor = np.array([factor_x, factor_y])
    converted_regions = []
    for r in regions:
        r_converted = np.around(r * factor)
        converted_regions.append(r_converted)

    return converted_regions, factor

def gen_full_segmentation_map(level_dims, regions, region_labels, label_map):
    """
    Generate a segmentation map with size of the total level dimensions
    """
    seg_map_full = Image.new('L', level_dims, color=0)
    draw = ImageDraw.Draw(seg_map_full)
    for idx, reg in enumerate(regions):
        #import pdb; pdb.set_trace()
        if check_label_exists(region_labels[idx], label_map):
            draw.polygon(reg.flatten().tolist(), fill=label_map[region_labels[idx]])
        else:
            print("[py-wsi warning]: region label not found, region not drawn.")
    #seg_map_full.show()
    return seg_map_full

def get_regions(path):
    ''' Parses the xml at the given path, assuming annotation format importable by ImageScope. '''
    xml = minidom.parse(path)
    # The first region marked is always the tumour delineation
    regions_ = xml.getElementsByTagName("Region")
    regions, region_labels = [], []
    for region in regions_:
        vertices = region.getElementsByTagName("Vertex")
        attribute = region.getElementsByTagName("Attribute")
        if len(attribute) > 0:
            r_label = attribute[0].attributes['Value'].value
        else:
            r_label = region.getAttribute('Text')
        region_labels.append(r_label)

        # Store x, y coordinates into a 2D array in format [x1, y1], [x2, y2], ...
        coords = np.zeros((len(vertices), 2))

        for i, vertex in enumerate(vertices):
            coords[i][0] = vertex.attributes['X'].value
            coords[i][1] = vertex.attributes['Y'].value

        regions.append(coords)
    return regions, region_labels

def patch_to_tile_size(patch_size, overlap):
    return patch_size - overlap*2

def tile_to_patch_size(tile_size, overlap):
    return tile_size + overlap*2

def sample_and_store_patches(file_name,
                             file_dir,
                             pixel_overlap,
                             env=False,
                             meta_env=False,
                             patch_size=512,
                             level=0,
                             xml_dir=False,
                             label_map={},
                             limit_bounds=True,
                             rows_per_txn=20,
                             db_location='',
                             prefix='',
                             storage_option='lmdb',
                             gen_segmentation_map=False):
    ''' Sample patches of specified size from .svs file.
        - file_name             name of whole slide image to sample from
        - file_dir              directory file is located in
        - pixel_overlap         pixels overlap on each side
        - env, meta_env         for LMDB only; environment variables
        - level                 0 is lowest resolution; level_count - 1 is highest
        - xml_dir               directory containing annotation XML files
        - label_map             dictionary mapping string labels to integers
        - rows_per_txn          how many patches to load into memory at once
        - storage_option        the patch storage option
        - gen_segmentation_map  if true, a binary segmentation map for each
                                patch is also stored

        Note: patch_size is the dimension of the sampled patches, NOT equivalent to openslide's definition
        of tile_size. This implementation was chosen to allow for more intuitive usage.
    '''

    tile_size = patch_to_tile_size(patch_size, pixel_overlap)
    slide = open_slide(file_dir + file_name)
    tiles = DeepZoomGenerator(slide,
                              tile_size=tile_size,
                              overlap=pixel_overlap,
                              limit_bounds=limit_bounds)

    level_dims = tiles.level_dimensions
    level_count = tiles.level_count
    if xml_dir:
        # Expect filename of XML annotations to match SVS file name
        regions, region_labels = get_regions(xml_dir + file_name[:-4] + ".xml")
        if gen_segmentation_map:
            # Convert region coordinates to level coordinates
            converted_regions, factor = transform_regions(regions, level_dims[level],
                                                          level_dims[level_count-1])
            seg_map_full = gen_full_segmentation_map(level_dims[level],
                                                     converted_regions, 
                                                     region_labels, 
                                                     label_map)


    if level >= tiles.level_count:
        print("[py-wsi error]: requested level does not exist. Number of slide levels: " + str(tiles.level_count))
        return 0
    x_tiles, y_tiles = tiles.level_tiles[level]

    x, y = 0, 0
    count, batch_count = 0, 0
    patches, coords, labels, seg_maps = [], [], [], []
    while y < y_tiles:
        while x < x_tiles:
            new_tile = np.array(tiles.get_tile(level, (x, y)), dtype=np.uint8)
            # OpenSlide calculates overlap in such a way that sometimes depending on the dimensions, edge
            # patches are smaller than the others. We will ignore such patches.
            if np.shape(new_tile) == (patch_size, patch_size, 3):
                patches.append(new_tile)
                coords.append(np.array([x, y]))
                count += 1

                # Calculate the patch label 
                if xml_dir:
                    # Convert coordinates to base level resolution
                    converted_coords = tiles.get_tile_coordinates(level, (x, y))[0]
                    # Default: one label for patch based on centre point
                    labels.append(generate_label(regions, region_labels, converted_coords, label_map))
                    # Optionally calculate gt patch for segmentation
                    if gen_segmentation_map:
                        # Coordinates on current level resolution
                        level_coords = np.around(converted_coords * factor)
                        # Segmentation map based on region label per patch
                        seg_maps.append(generate_segmentation_patch(seg_map_full,
                                                                 level_coords,
                                                                 patch_size)) 
            x += 1

        # To save memory, we will save data into the dbs every rows_per_txn rows. i.e., each transaction will commit
        # rows_per_txn rows of patches. Write after last row regardless. HDF5 does NOT follow
        # this convention due to efficiency.
        if (y % rows_per_txn == 0 and y != 0) or y == y_tiles-1:
            print(f"\nSaving tile rows {int((y-1)/rows_per_txn)*rows_per_txn} to {y}")
            if storage_option == 'disk':
                save_to_disk(db_location, patches, coords, file_name[:-4],
                             labels, seg_maps)
            elif storage_option == 'lmdb':
                # LMDB by default.
                save_in_lmdb(env, patches, coords, file_name[:-4], labels,
                             seg_maps)
            if storage_option != 'hdf5':
                del patches
                del coords
                del labels
                patches, coords, labels, seg_maps = [], [], [], [] # Reset right away.

        y += 1
        x = 0

    # Write to HDF5 files all in one go.
    if storage_option == 'hdf5':
        save_to_hdf5(db_location, patches, coords, file_name[:-4], labels)

    # Need to save tile dimensions if LMDB for retrieving patches by key.
    if storage_option == 'lmdb':
        save_meta_in_lmdb(meta_env, file_name[:-4], [x_tiles, y_tiles])

    return count

