# import modules
from .IdptCalibrationStack import IdptCalibrationStack


import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging
logger = logging.getLogger(__name__)


class IdptCalibrationModel(object):

    def __init__(self, collections, image_to_z):
        super(IdptCalibrationModel, self).__init__()

        self._particle_ids = None

        if not isinstance(image_to_z, list):
            if not isinstance(image_to_z, dict):
                raise TypeError("image_to_z must be a dictionary with keys image names and z coordinates "
                                "as values. Received type {}".format(type(image_to_z)))
            else:
                image_to_z = [image_to_z]

        if not isinstance(collections, list):
            collections = [collections]

        # Set the z height of each image using image_to_z mapper
        for collection, img_to_z in zip(collections, image_to_z):
            for image in collection.images.values():
                if image.filename not in img_to_z.keys():
                    raise ValueError("No z coordinate specified for image {}")
                else:
                    # set both the true_z and z value for each image and particle if the particle z-coord is None.
                    image.set_z(img_to_z[image.filename])

        # Create the calibration stacks
        self._create_stacks(*collections)

    def __len__(self):
        return len(self.calibration_stacks)

    def __repr__(self):
        class_ = 'GdpytCalibrationSet'
        repr_dict = {
            'Calibration stacks for particle IDs': list(self.calibration_stacks.keys())}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def _create_stacks(self, *collections):
        stacks = {}
        ids_in_collects = []
        for i, collection in enumerate(collections):
            # create a new collection (i.e. GdpytImageCollection) ID if multiple image collections are passed in.
            if i != 0:
                new_id_map = {}
                new_id = max(ids_in_collects) + 1

            # loop through all particle ID's and create a calibration stack class for each, then build stack layers.
            for image in collection.images.values():
                for particle in image.particles:

                    # if multiple collections, reassign a unique ID to each particle so stacks from different images
                    # between collections don't get mixed up.
                    if i != 0:
                        if particle.id not in new_id_map.keys():
                            new_id_map.update({particle.id: new_id})
                            particle.reset_id(new_id)
                            new_id += 1
                        else:
                            particle.reset_id(new_id_map[particle.id])

                    if particle.id not in stacks.keys():
                        # instantiate GdpytCalibrationStack class for each particle ID.
                        new_stack = IdptCalibrationStack(particle.id, particle.location)
                        new_stack.add_particle(particle)
                        stacks.update({particle.id: new_stack})
                    else:
                        stacks[particle.id].add_particle(particle)

                    ids_in_collects.append(particle.id)

            # once all the particles ID's have been assigned to a stack, build the calibration stack layers.
            for stack in stacks.values():
                stack.build_layers()

        # define the GdpytCalibrationSet's stacks.
        self._calibration_stacks = stacks

    def infer_z(self, infer_collection):
        return IdptImageInference(infer_collection, self)

    def update_particle_ids(self):
        self._particle_ids = list(self.calibration_stacks.keys())

    @property
    def calibration_stacks(self):
        return self._calibration_stacks

    @property
    def particle_ids(self):
        self.update_particle_ids()
        return self._particle_ids


class IdptImageInference(object):

    def __init__(self, infer_collection, calib_set):
        self.collection = infer_collection
        assert isinstance(calib_set, IdptCalibrationModel)
        self.calib_set = calib_set

    def _cross_correlation_inference(self, function, use_stack=None):

        if function.lower() not in ['sknccorr']:
            raise ValueError("{} is not implemented or a valid function".format(function))

        if use_stack == 'nearest':
            baseline_locations = []
            for stack in self.calib_set.calibration_stacks.values():
                baseline_locations.append(pd.DataFrame({'x': stack.location[0], 'y': stack.location[1]},
                                                       index=[stack.id]))
            baseline_locations = pd.concat(baseline_locations).sort_index()
            nneigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(baseline_locations.values)

        for image in self.collection.images.values():
            logger.info("Inferring image {}".format(image.filename))
            for particle in image.particles:
                if use_stack is not None:
                    if use_stack == 'nearest':
                        distances, indices = nneigh.kneighbors(np.array([particle.location[0], particle.location[1]]))
                        for distance, idx in zip(distances, indices):
                            matching_stack_id = baseline_locations.index.values[idx.squeeze()]
                            stack = self.calib_set.calibration_stacks[matching_stack_id]
                    else:
                        stack = self.calib_set.calibration_stacks[use_stack]
                elif particle.id in self.calib_set.particle_ids:
                    stack = self.calib_set.calibration_stacks[particle.id]
                else:
                    print("Test particle ID {} not found in calib set".format(particle.id))
                    continue

                # set the stack ID used for z-inference
                particle.set_inference_stack_id(stack.id)

                # infer z
                stack.infer_z(particle, function=function)

    def sknccorr(self, use_stack=None):
        self._cross_correlation_inference('sknccorr', use_stack=use_stack)

    @property
    def infer_sub_image(self):
        return self._infer_sub_image