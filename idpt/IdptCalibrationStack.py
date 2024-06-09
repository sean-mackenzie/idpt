# IdptCalibrationStack.py

from .IdptParticle import IdptParticle
from idpt.utils.correlation import get_similarity_function, sk_norm_cross_correlation, correlate_against_stack
from idpt.utils.correlation import localize_discrete, localize_subresolution, parabolic_interpolation
from idpt.utils.subresolution import fit_2d_gaussian_on_corr

from collections import OrderedDict
import numpy as np

import logging

logger = logging.getLogger(__name__)


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
        class_ = 'GdpytCalibrationStack'
        min_z = min(list(self.layers.keys()))
        max_z = max(list(self.layers.keys()))
        repr_dict = {'Particle ID': self.id,
                     'Location (x, y)': self.location,
                     'Particle bounding box dimensions': self.shape,
                     'Number of layers': len(self),
                     'Min. and max. z coordinate': [min_z, max_z]}
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def add_particle(self, particle):
        assert isinstance(particle, IdptParticle)
        self._particles.append(particle)

    def build_layers(self):
        # uniformize template size and center
        print("skipping uniformize and center templates. Need to double-check")
        # self._uniformize_and_center()

        # get list of z-coordinates and template images
        z = []
        templates = []
        for particle in self._particles:
            z.append(particle.z)
            templates.append(particle.get_template())

        # create ordered dict and sort by z-coord
        layers = OrderedDict()
        for z, template in sorted(zip(z, templates), key=lambda k: k[0]):
            layers.update({z: template})

        # instantiate layers attribute
        self._layers = layers

    """
    def _uniformize_and_center(self):
        # Find biggest bounding box
        w_max, h_max = (0, 0)
        for particle in self._particles:
            w, h = (particle.bbox[2], particle.bbox[3])
            if w > w_max:
                w_max = w
            if h > h_max:
                h_max = h

        for particle in self._particles:
            logger.debug('Stack resize bbox: {}'.format((w_max, h_max)))
            particle.resize_bbox(w_max, h_max)

        self._shape = (w_max, h_max)
    """

    def get_layers(self, range_z):
        # if no range is supplied, get all layers
        if range_z is not None:
            raise ValueError("Not sure what this is")
        else:
            return self._layers

    """
    def infer_z_old(self, particle, function):
        # get array of z-coords and image templates
        z_calib, temp_calib = np.array(list(self.layers.keys())), np.array(list(self.layers.values()))

        # if/elif function to pass the correct cross-correlation method and optimum function
        if function == 'sknccorr':
            sim_func = sk_norm_cross_correlation
            optim = np.argmax
        else:
            raise ValueError("Unknown similarity function {}".format(function))

        # perform the cross-correlation against each image in the calibration stack and append the results to a list
        sim = []
        res = []
        match_location = []
        match_localization = []

        # 3D template matching
        # result = match_template(temp_calib_3d, particle.template[:, :, np.newaxis])

        for c_temp in temp_calib:
            # similarity, xm, ym, xg, yg = sim_func(c_temp, particle.template)
            result = sim_func(c_temp, particle.template)
            res.append(result)
            sim.append(np.max(result))
            # match_location.append([xm, ym])
            # match_localization.append([xg, yg])

        sim = np.array(sim)
        max_idx = optim(sim)
        particle.set_cm(sim[max_idx])

        # get the correlation map where peak correlation was found
        result = res[max_idx]
        res_length = np.floor(result.shape[0] / 2)

        # x,y coordinates in the image space where the highest correlation was found
        ij = np.unravel_index(np.argmax(result), result.shape)
        xmt, ymt = ij[::-1]
        xm = res_length - xmt
        ym = res_length - ymt

        # set template matching location according to particle location
        mlx, mly = xm, ym
        if mlx is not None and mly is not None:
            mlx, mly = particle.location[0] + mlx, particle.location[1] + mly
        particle.set_match_location([mlx, mly])

        # sub-pixel localization
        xg, yg = None, None
        result = result - np.min(result)
        pad_width = 0
        # result = np.pad(result, pad_width)

        locate_subpix = True
        if locate_subpix:
            if np.size(result) > 5:
                xgt, ygt = fit_2d_gaussian_on_corr(result, xmt + pad_width, ymt + pad_width)
                if xgt is not None and ygt is not None:
                    xg = res_length - xgt
                    yg = res_length - ygt
                    xg, yg = particle.location[0] + xg, particle.location[1] + yg

        particle.set_match_localization([xg, yg])

        # discrete
        particle.set_z(z_calib[optim(sim)])
        particle.set_max_sim(sim[max_idx])

        # subresolution
        z_interp, sim_interp = parabolic_interpolation(z_calib, sim, max_idx)
        particle.set_z(z_interp[optim(sim_interp)])
        particle.set_max_sim(sim_interp[optim(sim_interp)])
    """

    def infer_z(self, particle, function):
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

    def reset_id(self, new_id):
        assert isinstance(new_id, int)
        self._id = new_id

        for particle in self.particles:
            particle.reset_id(new_id)

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