# IdptImage.py

from os.path import join

from skimage import io
from skimage.filters import median, gaussian, roberts, sobel

from skimage.exposure import rescale_intensity
from skimage.measure import regionprops_table

from scipy.signal import convolve2d

import numpy as np
import numpy.ma as ma
import pandas as pd
from idpt.utils.particle_identification import apply_threshold, identify_contours, pad_and_center_region
from .IdptParticle import IdptParticle
from os.path import isfile, basename
import time
import logging

logger = logging.getLogger()


class IdptImage(object):

    def __init__(self, path, frame=None):
        super(IdptImage, self).__init__()

        if not isfile(path):
            raise ValueError("{} is not a valid file".format(path))

        self._filepath = path
        self._filename = basename(path)
        self.frame = frame

        self._raw = None  # raw image (can be modified)
        self._original = None  # original image (unmodified)
        self._subbg = None  # background subtracted image

        # Load the image. This sets the ._raw attribute
        self.load()

        # Filtered image. This attribute is assigned by using the filter_image method
        self._filtered = None
        self._processing_stats = None
        self._masked = None

        # Particles: dictionary {particle_id: Particle object}
        # This dictionary is filled up with the identify_particles method
        self._particles = []
        self._z = None

        # contour stats
        self.regionprops_table = None
        self._mean_contour_area = None
        self._std_contour_area = None

    def load(self):
        img = io.imread(self._filepath, plugin='tifffile')

        # check if image is a stack
        if len(np.shape(img)) > 2:  # image is a stack
            if np.shape(img)[0] < np.shape(img)[2]:
                img = np.rint(np.mean(img, axis=0, dtype=float)).astype(img.dtype)
            else:
                img = np.rint(np.mean(img, axis=2, dtype=float)).astype(img.dtype)

        self._original = img
        self._raw = img

    def crop_image(self, cropspecs):
        valid_crops = ['xmin', 'xmax', 'ymin', 'ymax', 'pad']
        for crop_func in cropspecs.keys():
            if crop_func not in valid_crops:
                raise ValueError("{} is not a valid crop dimension."
                                 "Use: {}".format(crop_func, valid_crops))
        self._raw = self._original[cropspecs['ymin']:cropspecs['ymax'],
                    cropspecs['xmin']:cropspecs['xmax']]
        if 'pad' in cropspecs.keys():
            self._raw = np.pad(self._raw, pad_width=cropspecs['pad'], mode='constant',
                               constant_values=np.min(self._raw))

    def subtract_background(self, background_subtraction, background_img):
        pass

    def preprocess_image(self, filterspecs):
        self._filtered = self._raw

    def _add_particle(self, id_, contour, bbox, particle_mask, location, template_use_raw):
        if template_use_raw:
            img = self._raw
        else:
            img = self._filtered
        self._particles.append(IdptParticle(img, id_, contour, bbox,
                                            particle_mask_on_image=particle_mask,
                                            location=location,
                                            frame=self.frame,
                                            ))

    def identify_particles(self, collection, particle_id_image, thresh_specs,
                           min_size, max_size, padding, template_use_raw):

        if collection.baseline_regions is not None and particle_id_image is not None:
            particle_mask = collection.baseline_particle_mask
            regions = collection.baseline_regions
            all_contour_coords = collection.baseline_all_contour_coords
            regionprops_data = collection.baseline_regionprops_data
        else:
            show_threshold = False
            # b/c static templates
            particle_mask = apply_threshold(img=particle_id_image,
                                            parameter=thresh_specs,
                                            show_threshold=show_threshold).astype(np.uint16)

            # identify particles
            label_image, regions, all_contour_coords = identify_contours(particle_mask, self.filtered)

            # store the regionprops table
            regionprops_data = regionprops_table(label_image, self.filtered,
                                                 properties=['label', 'area', 'bbox', 'centroid', 'weighted_centroid',
                                                             'local_centroid',
                                                             'weighted_local_centroid', 'max_intensity',
                                                             'mean_intensity',
                                                             'minor_axis_length', 'major_axis_length'])
            regionprops_data = pd.DataFrame(regionprops_data)
            self.regionprops_table = regionprops_data

            # set collection baseline regions
            # if particle_id_image is not None:
            collection.set_baseline_particle_mask(particle_mask)
            collection.set_baseline_regions(regions)
            collection.set_baseline_all_contour_coords(all_contour_coords)
            collection.set_baseline_regionprops_data(regionprops_data)

            # ---

        # filters regions (contours)
        skipped_contours = []
        passing_ids = []
        contour_areas = []
        id_ = 0

        # ---

        # Sort contours and bboxes by x-coordinate:
        for region, contour_coords in sorted(zip(regions, all_contour_coords), key=lambda x: x[0].centroid[1]):

            # filter on area
            area = region.area
            if area < min_size or area > max_size:
                if self.frame == 0:
                    logger.warning("Region skipped b/c area {} < {} | area {} > {}".format(area, min_size, area, max_size))
                skipped_contours.append(region.label)
                continue

            aspect_ratio_threshold = 3
            if min_size > 5:
                aspect_ratio = region.major_axis_length / region.minor_axis_length
                if aspect_ratio > aspect_ratio_threshold:
                    if self.frame == 0:
                        logger.warning(
                            "Region skipped b/c aspect ratio = {} > {}.".format(aspect_ratio, aspect_ratio_threshold))
                    skipped_contours.append(region.label)
                    continue

            # adjust the bounding box (bbox) to work with GdpytParticle (note: x0, y0, w0, h0 = self.bbox)
            min_row, min_col, max_row, max_col = region.bbox
            bbox = (min_col, min_row, max_col - min_col, max_row - min_row)

            cX = int(
                np.round(regionprops_data[regionprops_data['label'] == region.label]['weighted_centroid-1'].item(), 0))
            cY = int(
                np.round(regionprops_data[regionprops_data['label'] == region.label]['weighted_centroid-0'].item(), 0))

            # pad and center template
            bbox, bbox_center = pad_and_center_region(cX, cY, bbox, padding=padding)

            # discard contours that are too close to the
            # image borders to include the desired padding
            filter_borders = True
            if filter_borders:
                if bbox[0] - padding * 0.1 < 1 or bbox[1] - padding * 0.1 < 1:
                    # NOTE: the constant 0.1 used to be 0.5
                    skipped_contours.append(region.label)
                    if self.frame == 0:
                        print("FIRST FILTER: Skipped because template + padding near the image borders")
                    continue
                elif bbox[0] + bbox[2] + padding * 0.1 >= self.shape[1] or bbox[1] + bbox[3] + padding * 0.1 >= \
                        self.shape[0]:
                    # NOTE: the constant 0.1 used to be 0.5
                    skipped_contours.append(region.label)
                    if self.frame == 0:
                        print("SECOND FILTER: Skipped because template + padding near the image borders")
                    continue

            # Add data for contour inspection
            contour_areas.append(area)
            passing_ids.append(id_)

            # Add particle
            self._add_particle(id_, contour_coords, bbox, particle_mask,
                               location=(cX, cY), template_use_raw=template_use_raw)
            id_ = id_ + 1

        # Calculate contour statistics
        if len(contour_areas) > 0:
            min_contour_area = np.min(contour_areas)
            max_contour_area = np.max(contour_areas)
            mean_contour_area = np.mean(contour_areas)
            std_contour_area = np.round(np.std(contour_areas), 1)
            if self.frame == 0:
                if min_contour_area < min_size * 1.5:
                    print("Min. contour area [input, measured] = [{}, {}]".format(min_size, min_contour_area))
                    print("Mean contour area: {} +/- {}".format(mean_contour_area, std_contour_area * 2))
                if max_contour_area > max_size * 0.75:
                    print("Max. contour area [input, measured] = [{}, {}]".format(max_size, max_contour_area))
                    print("Mean contour area: {} +/- {}".format(mean_contour_area, std_contour_area * 2))

            # set image attributes
            self._masked = particle_mask
            self._mean_contour_area = mean_contour_area
            self._std_contour_area = std_contour_area

    def set_z(self, z):
        assert isinstance(z, float)
        # set the z-value of the image
        self._z = z
        # If the image is set to be at a certain height, all the particles' true_z are assigned that height
        for particle in self.particles:
            # only set particle z-coordinate if it hasn't already been set
            if particle.z is None:
                particle.set_z(z)


    @property
    def filename(self):
        return self._filename

    @property
    def filepath(self):
        return self._filepath

    @property
    def subbg(self):
        return self._subbg

    @property
    def filtered(self):
        return self._filtered

    @property
    def processed(self):
        return self._filtered

    @property
    def masked(self):
        return self._masked

    @property
    def original(self):
        return self._original

    @property
    def particles(self):
        return self._particles

    @property
    def raw(self):
        return self._raw

    @property
    def shape(self):
        return self.raw.shape

    @property
    def z(self):
        return self._z