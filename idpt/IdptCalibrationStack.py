# IdptCalibrationStack.py

from .IdptParticle import IdptParticle
from idpt.utils.correlation import get_similarity_function, correlate_against_stack
from idpt.utils.correlation import localize_discrete, localize_subresolution
from collections import OrderedDict
import numpy as np


class IdptCalibrationStack(object):

    def __init__(self, particle_id, location):
        super(IdptCalibrationStack, self).__init__()
        self._id = particle_id
        self._location = location
        self._layers = OrderedDict()
        self._particles = []
        self._shape = None

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, item):
        if isinstance(item, int):
            key = list(self.layers.keys())[item]
            return key, self.layers[key]
        else:
            return item, self.layers[item]

    def __repr__(self):
        class_ = 'IdptCalibrationStack'
        min_z = min(list(self.layers.keys()))
        max_z = max(list(self.layers.keys()))
        repr_dict = {'ID': self.id,
                     'Baseline position': self.location,
                     'Sub-image size': self.shape,
                     'Z-depth': len(self),
                     'Z-bounds': [min_z, max_z]}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def add_particle(self, particle):
        assert isinstance(particle, IdptParticle)
        self._particles.append(particle)

    def build_layers(self):
        # get list of z-coordinates and template images
        z = []
        templates = []
        for particle in self._particles:
            z.append(particle.z)
            templates.append(particle.get_template())

        # create ordered dict and sort by z-coordinate
        layers = OrderedDict()
        for z, template in sorted(zip(z, templates), key=lambda k: k[0]):
            layers.update({z: template})

        # instantiate layers attribute
        self._layers = layers

    def infer_z(self, particle, function):
        """

        :param particle:
        :param function:
        :return:
        """
        # stack of z-positions and templates
        z_calib, calib_stack = np.array(list(self.layers.keys())), np.array(list(self.layers.values()))

        # similarity function, optimization function
        sim_func, optim = get_similarity_function(function)

        # cross-correlate template with calibration stack
        idx_peak, similarity_stack, response_stack = correlate_against_stack(template=particle.template,
                                                                             stack=calib_stack,
                                                                             sim_func=function)
        # peak similarity corresponds to x-, y-, z-shift
        sim_discrete, dx_discrete, dy_discrete, z_discrete = localize_discrete(idx_peak, similarity_stack,
                                                                               response_stack, z_calib)
        # set particle's discrete position estimation
        particle.set_localized_discrete_position(sim_discrete, dx_discrete, dy_discrete, z_discrete)

        # interpolate in x-y and z to refine position below image resolution
        sim_sub, dx_sub, dy_sub, z_sub = localize_subresolution(idx_peak, similarity_stack,
                                                                response_stack, z_calib, optim)
        # set particle's sub-resolution position estimation
        particle.set_localized_subresolution_position(sim_sub, dx_sub, dy_sub, z_sub)

    @property
    def id(self):
        return self._id

    @property
    def location(self):
        return self._location

    @property
    def layers(self):
        return self._layers

    @property
    def shape(self):
        return self._shape

    @property
    def particles(self):
        return self._particles