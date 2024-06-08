# IdptImageCollection.py

from os.path import join, isdir
from os import listdir
import re
from collections import OrderedDict
import logging
import random

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

from .IdptCalibrationModel import IdptCalibrationModel
from .IdptImage import IdptImage


class IdptImageCollection(object):

    def __init__(self,
                 image_collection_type,
                 image_path,
                 image_file_type,
                 image_base_string,
                 calibration_z_step_size,
                 image_subset,
                 baseline_image,
                 hard_baseline,
                 cropping,
                 preprocessing,
                 thresholding,
                 background_subtraction,
                 min_particle_area,
                 max_particle_area,
                 template_padding,
                 same_id_threshold,
                 stacks_use_raw,
                 xy_displacement,
                 particle_id_image=None):
        super(IdptImageCollection, self).__init__()

        if not isdir(image_path):
            raise ValueError("Specified image path {} does not exist".format(image_path))

        # properties of the image collection
        self.image_collection_type = image_collection_type
        self.image_path = image_path
        self.image_file_type = image_file_type
        self.image_base_string = image_base_string
        self.calibration_z_step_size = calibration_z_step_size

        # image processing filters
        self.min_particle_size = min_particle_area
        self.max_particle_size = max_particle_area
        self.same_id_threshold = same_id_threshold

        # toggles for calibration stacks and inference
        self.baseline = baseline_image
        if particle_id_image is not None:
            self.particle_id_image = particle_id_image
        else:
            self.particle_id_image = baseline_image
        self.template_padding = template_padding
        self.hard_baseline = hard_baseline
        self.xy_displacement = xy_displacement
        self.stacks_use_raw = stacks_use_raw

        # image processing params
        self.cropping = cropping
        self.background_subtraction = background_subtraction
        self.preprocessing = preprocessing
        self.thresholding = thresholding

        # not known if necessary
        self.baseline_particle_mask = None
        self.baseline_regions = None
        self.baseline_all_contour_coords = None
        self.baseline_regionprops_data = None
        # not known if necessary

        # Will get defined
        self.files = None
        self.images = None

        # add images
        exclude = []
        self.get_exclusion_subset(exclude=exclude, subset=image_subset)
        self.find_files(exclude=exclude)
        self.add_images()

        # process images
        self.crop_images()
        self.subtract_background()
        self.preprocess_images()
        self.identify_particles_in_images()
        self.uniformize_particle_ids(baseline=self.baseline, uv=self.xy_displacement)

    def get_exclusion_subset(self, exclude, subset):
        if subset is None:
            pass
        else:
            base_string = self.image_base_string
            all_files = listdir(self.image_path)
            save_files = []

            for file in all_files:
                if file.endswith(self.image_file_type):
                    if file in exclude:
                        continue
                    save_files.append(file)

            # if subset is an integer, this indicates the total number of files to include.
            #   Note: the files are randomly selected from the collection.
            if len(subset) == 1:
                random_files = [rf for rf in random.sample(set(save_files), subset[0])]
                for f in save_files:
                    if f not in random_files:
                        exclude.append(f)

            # if subset is a two element list, this indicates a start and stop range for z-values.
            elif len(subset) == 2:
                start = subset[0]
                stop = subset[1]
                for f in save_files:
                    search_string = base_string + '(.*)' + self.image_file_type
                    file_index = float(re.search(search_string, f).group(1))
                    if file_index < start or file_index > stop:
                        exclude.append(f)

            # if subset is a three element list, this indicates a start and stop z-range and sampling rate.
            elif len(subset) == 3:

                protected_files = []
                # we always want to include the baseline or particle_id_image in the image collection.
                if self.baseline is not None:
                    if isinstance(self.baseline, IdptCalibrationModel):
                        pass
                    else:
                        protected_files.append(self.baseline)
                if self.particle_id_image is not None:
                    protected_files.append(self.particle_id_image)

                start = subset[0]
                stop = subset[1]
                subset_files = []
                subset_index = []

                # get files within subset z-coordinates
                for f in save_files:
                    search_string = base_string + '(.*)' + self.image_file_type
                    file_index = float(re.search(search_string, f).group(1))
                    if file_index >= start and file_index <= stop:
                        subset_files.append(f)
                        subset_index.append(file_index)

                # sort the zipped list of files and indices (floats for the file's z-coordinate)
                sorted_subset = sorted(list(zip(subset_files, subset_index)), key=lambda x: x[1])

                print(sorted_subset)

                # sample the list according to the third element in subset
                sorted_files, sorted_indices = map(list, zip(*sorted_subset))
                n_sampling = subset[2]
                sampled_sorted_subset_files = sorted_files[::n_sampling]

                # append files not sampled to exclude
                for f in save_files:
                    if f not in sampled_sorted_subset_files + protected_files:
                        exclude.append(f)

            else:
                raise ValueError("Collecting multiple subsets is not implemented at the moment.")

    def find_files(self, exclude=[]):
        all_files = listdir(self.image_path)
        number_of_files = 0
        save_files = []
        for file in all_files:
            if file.endswith(self.image_file_type):
                if file in exclude:
                    number_of_files += 1
                    continue
                save_files.append(file)
                number_of_files += 1

        save_files = sorted(save_files,
                            key=lambda filename: float(filename.split(self.image_base_string)[-1].split('.')[0]))
        logger.warning(
            "Found {} files with filetype {} in folder {}".format(len(save_files), self.image_file_type,
                                                                  self.image_path))
        self.files = save_files

    def add_images(self):
        images = OrderedDict()
        frame = 0
        for file in self.files:
            img = IdptImage(join(self.image_path, file), frame=frame)
            images.update({img.filename: img})
            frame += 1
        self.images = images

    def crop_images(self):
        pass

    def subtract_background(self):
        pass

    def preprocess_images(self):
        for image in self.images.values():
            image.preprocess_image(self.preprocessing)

    def identify_particles_in_images(self):
        if self.image_collection_type == 'calibration':
            # b/c static templates
            particle_identification_image = self.images[self.baseline].filtered

        else:
            if isinstance(self.baseline, IdptCalibrationModel):
                particle_identification_image = self.images[self.particle_id_image].filtered
            else:
                particle_identification_image = self.images[self.baseline].filtered

        for i, image in enumerate(self.images.values()):
            image.identify_particles(collection=self,
                                     particle_id_image=particle_identification_image,
                                     thresh_specs=self.thresholding,
                                     min_size=self.min_particle_size,
                                     max_size=self.max_particle_size,
                                     padding=self.template_padding,
                                     template_use_raw=self.stacks_use_raw,
                                     )
            if i == 0:
                logger.info("Identified {} particles on image {}".format(len(image.particles), image.filename))

    def uniformize_particle_ids(self, baseline=None, baseline_img=None, uv=None):
        if uv is None:
            uv = [[0, 0]]

        baseline_locations = []
        # If a calibration set is given as the baseline, the particle IDs in this collection are assigned based on
        # the location and ID of the calibration set. This should be done when the collection contains target images
        if baseline is not None:
            if isinstance(baseline, IdptCalibrationModel):
                for stack in baseline.calibration_stacks.values():
                    baseline_locations.append(pd.DataFrame({'x': stack.location[0], 'y': stack.location[1]},
                                                           index=[stack.id]))
                skip_first_img = False

            elif isinstance(baseline, IdptImage):
                baseline_img = baseline

                for particle in baseline_img.particles:
                    baseline_locations.append(pd.DataFrame({'x': particle.location[0], 'y': particle.location[1]},
                                                           index=[particle.id]))
                skip_first_img = False
            elif isinstance(baseline, str):
                baseline_img = self.images[baseline]
                for particle in baseline_img.particles:
                    baseline_locations.append(pd.DataFrame({'x': particle.location[0], 'y': particle.location[1]},
                                                           index=[particle.id]))
                skip_first_img = False
            else:
                raise TypeError("Invalid type for baseline")

        # If no baseline is given, the particle IDs are assigned based on the IDs and location of the particles in the
        # baseline_img or else the first image
        else:
            index = 0
            if baseline_img is not None:
                index = self.files.index(baseline_img)
            baseline_img = self.files[index]
            baseline_img = self.images[baseline_img]

            for particle in baseline_img.particles:
                baseline_locations.append(pd.DataFrame({'x': particle.location[0], 'y': particle.location[1]},
                                                       index=[particle.id]))

            skip_first_img = True

        if len(baseline_locations) == 0:
            baseline_locations = pd.DataFrame()
            next_id = None
        else:
            baseline_locations = pd.concat(baseline_locations).sort_index()
            baseline_locations['x'] = baseline_locations['x'] + uv[0][0]
            baseline_locations['y'] = baseline_locations['y'] + uv[0][1]
            # The next particle that can't be matched to a particle in the baseline gets this id
            next_id = len(baseline_locations)

        for i, file in enumerate(self.files):
            if (i == 0) and skip_first_img:
                continue
            image = self.images[file]
            # Convert to list because ordering is important
            particles = [particle for particle in image.particles]
            locations = [list(p.location) for p in particles]

            if len(locations) == 0:
                continue
            if baseline_locations.empty:
                dfs = [pd.DataFrame({'x': p.location[0], 'y': p.location[1]}, index=[p.id]) for p in particles]
                baseline_locations = pd.concat(dfs)
                next_id = len(baseline_locations)
                continue

            # NearestNeighbors(x+u,y+v): previous location (x,y) + displacement guess (u,v)
            nneigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(baseline_locations.values)
            # NOTE: the displacement guess (u,v) could incorporate a "time" variable (image number or time data)
            # such that [u_i,v_i] = [u(t), v(t)] in order to better match time-dependent or periodic displacements.

            distances, indices = nneigh.kneighbors(np.array(locations))

            remove_p_not_in_calib = []
            for distance, idx, particle in zip(distances, indices, particles):
                # If the particle is close enough to a particle of the baseline, give that particle the same ID as the
                # particle in the baseline
                if distance < self.same_id_threshold:

                    particle.set_id(baseline_locations.index.values[idx.squeeze()])

                    # assign the baseline coordinates (x,y) to the matched particle coordinates (x,y)
                    baseline_locations.loc[particle.id, ('x', 'y')] = (particle.location[0], particle.location[1])
                    # the baseline is essentially the "current" location for that particle ID and after each
                    # successful identification, we update the "current" location of that particle ID.

                else:
                    # If the particle is not in the baseline, we may remove it via two methods:
                    #   1. if the baseline is a CalibrationSet, as including will significantly reduce accuracy.
                    #   2. if we designate a "hard baseline" where we don't want to add new particles.

                    # filter if not in CalibrationSet baseline:
                    if isinstance(baseline, IdptCalibrationModel):
                        remove_p_not_in_calib.append(particle)
                        if i == 0:
                            logger.warning(
                                "Removed particle {} at location {} b/c "
                                "not in Calibration Set baseline".format(particle.id, particle.location))
                        continue

                    # filter if not in "hard baseline":
                    elif self.hard_baseline is True:
                        remove_p_not_in_calib.append(particle)
                        logger.warning(
                            "Removed particle {} at location {} b/c not in hard baseline".format(particle.id,
                                                                                                 particle.location))
                        continue

                    # else, assign it a new, non-existent id and add it to the baseline for subsequent images
                    logger.warning("File {}: New IDs: {}".format(file, next_id))
                    particle.set_id(next_id)
                    assert (next_id not in baseline_locations.index)
                    baseline_locations = baseline_locations.append(
                        pd.DataFrame({'x': particle.location[0], 'y': particle.location[1]},
                                     index=[particle.id]))
                    next_id += 1
            for p in remove_p_not_in_calib:
                image.particles.remove(p)

    def set_baseline_particle_mask(self, particle_mask):
        self.baseline_particle_mask = particle_mask

    def set_baseline_regions(self, regions):
        self.baseline_regions = regions

    def set_baseline_all_contour_coords(self, all_contour_coords):
        self.baseline_all_contour_coords = all_contour_coords

    def set_baseline_regionprops_data(self, regionprops_data):
        self.baseline_regionprops_data = regionprops_data

    def create_calibration(self, name_to_z, template_padding=0):
        return IdptCalibrationModel(self, name_to_z, template_padding=template_padding)

    def infer_z(self, calib_set):
        assert isinstance(calib_set, IdptCalibrationModel)
        return calib_set.infer_z(self)