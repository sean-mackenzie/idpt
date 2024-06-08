

import numpy as np
import logging
logger = logging.getLogger(__name__)


class IdptParticle(object):

    def __init__(self, image, id_, contour, bbox,
                 particle_mask_on_image, location, frame):
        super(IdptParticle, self).__init__()
        self._id = int(id_)
        self.frame = frame
        self._image = image
        self._contour = contour
        self._bbox = bbox
        self._mask_on_image = particle_mask_on_image
        self._location = location

        self._template = None
        self.match_location = None
        self.match_localization = None
        self._location_on_template = None
        self._mask_on_template = None
        self._template_contour = None
        self._in_images = None
        self.inference_stack_id = None
        self._cm = None
        self._max_sim = None
        self._z = None
        self._z_default = None

        # set the _template, _location_on_template, and _template_contour attributes.
        self._create_template(bbox=bbox)

    def _create_template(self, bbox=None):
        image = self._image

        # if no bbox is passed in, use the particle.bbox variable
        if bbox is None:
            x0, y0, w0, h0 = self.bbox
            x, y, w, h = self.bbox
        else:
            x0, y0, w0, h0 = bbox
            x, y, w, h = bbox

        orig_template = image[y: y + h, x: x + w]

        # adjust the bounding box so it doesn't exceed the image bounds
        pad_x_m, pad_x_p, pad_y_m, pad_y_p = 0, 0, 0, 0

        if y + h > image.shape[0]:
            pad_y_p = y + h - image.shape[0]
        if y < 0:
            pad_y_m = - y
            h = y + h
            y = 0
        if x + w > image.shape[1]:
            pad_x_p = x + w - image.shape[1]
        if x < 0:
            pad_x_m = - x
            w = x + w
            x = 0
        pad_x = (pad_x_m, pad_x_p)
        pad_y = (pad_y_m, pad_y_p)

        # if no padding is necessary, instantiate particle variables
        if (pad_x == (0, 0)) and (pad_y == (0, 0)):

            # new method using Silvan's framework
            x0, y0, w0, h0 = self.bbox
            orig_template = image[y: y + h, x: x + w]
            # set the template
            self._template = image[y: y + h, x: x + w]
            #  [y - 1: y + h - 1, x - 1: x + w - 1] [y + 1: y + h + 1, x + 1: x + w + 1]

            # set mask on template
            self._mask_on_template = self.mask_on_image[y: y + h, x: x + w]

            # set particle center location on template
            self._location_on_template = (self.location[0] - x, self.location[1] - y)

            # define the contour in template coordinates
            contr = np.squeeze(self.contour)
            self._template_contour = np.array([contr[:, 0] - x, contr[:, 1] - y]).T

            # check if template is all nans
            array_nans = np.isnan(self.template)
            count_nans = np.sum(array_nans)

        else:
            # from Silvan's original code
            tempy = image[y: y + h, x: x + w]

            template = np.pad(tempy.astype(np.float), (pad_y, pad_x),
                              'constant', constant_values=np.median(tempy))
            # changed from: "'constant', np.nan)" on 7/23/2022

            # the below are my additions
            # set the template
            self._template = template  # image[yl:yr, xl:xr]

            # set mask on template
            self._mask_on_template = np.pad(self.mask_on_image[y: y + h, x: x + w].astype(np.float), (pad_y, pad_x),
                                            'constant', constant_values=0)  # changed from np.nan, 7/23/2022

            # set particle center location on template
            self._location_on_template = (np.shape(template)[0] // 2, np.shape(template)[1] // 2)

            # define the contour in template coordinates
            contr = np.squeeze(self.contour)
            self._template_contour = np.array([contr[:, 0] - x, contr[:, 1] - y]).T

            # check if template is all nans
            array_nans = np.isnan(self.template)
            count_nans = np.sum(array_nans)

        return self.template

    def add_particle_in_image(self, img_id):
        self._in_images = img_id

    def _dilated_bbox(self, dilation=None, dims=None):
        if dims is None:
            w, h = self.bbox[2], self.bbox[3]
        else:
            w, h = dims
        if dilation is None:
            return self.bbox
        elif isinstance(dilation, tuple):
            assert len(dilation) == 2
            dil_x, dil_y = dilation
        elif isinstance(dilation, float) or isinstance(dilation, int):
            dil_x = dilation
            dil_y = dilation
        else:
            raise TypeError("Wrong type for dilation (Received {})".format(type(dilation)))

        wl, ht = int(w * dil_x / 2), int(h * dil_y / 2)
        top_corner = np.array(self.location).astype(int) - np.array([wl, ht])
        dilated_bbox = (top_corner[0], top_corner[1], int(w * dil_x), int(h * dil_y))
        return dilated_bbox

    def _resized_bbox(self, resize=None):
        if resize is None:
            return self.bbox
        else:
            w, h = resize
            wl, ht = int(np.floor(w / 2)), int(np.floor(h / 2))
            top_corner = np.array(self.location).astype(int) - np.array([wl, ht])
            return top_corner[0], top_corner[1], w, h

    def resize_bbox(self, w, h):
        self._bbox = self._resized_bbox(resize=(w, h))
        self._create_template()

    def get_template(self, dilation=None, resize=None):
        if dilation is None and resize is None:
            return self._create_template()
        elif dilation is not None and resize is None:
            dil_bbox = self._dilated_bbox(dilation=dilation)
            return self._create_template(bbox=dil_bbox)
        elif dilation is None and resize is not None:
            resized_bbox = self._resized_bbox(resize)
            return self._create_template(bbox=resized_bbox)
        else:
            resized_bbox = self._resized_bbox(resize=resize)
            dil_bbox = self._dilated_bbox(dilation=dilation, dims=resized_bbox[2:])
            return self._create_template(bbox=dil_bbox)

    def set_id(self, id_):
        self._id = id_

    def reset_id(self, new_id):
        assert isinstance(new_id, int)
        logger.warning("Particle ID {}: Reset ID to {}".format(self.id, new_id))
        self._id = new_id

    def set_inference_stack_id(self, stack):
        self.inference_stack_id = stack

    def _set_location(self, location):
        assert len(location) == 2
        self._location = location

    def set_match_location(self, match_loc):
        self.match_location = (match_loc[0], match_loc[1])

    def set_match_localization(self, match_loc):
        self.match_localization = (match_loc[0], match_loc[1])

    def set_cm(self, c_measured):
        assert isinstance(c_measured, float)
        self._cm = c_measured

    def set_max_sim(self, sim):
        self._max_sim = sim

    def set_z(self, z):
        assert isinstance(z, float)
        self._z = z
        # The value originally received is stored in a separate argument
        if self._z_default is None:
            self._z_default = z

    @property
    def bbox(self):
        return self._bbox

    @property
    def contour(self):
        return self._contour

    @property
    def id(self):
        return self._id

    @property
    def in_images(self):
        return self._in_images

    @property
    def location(self):
        """
        Notes: the location is in index-array coordinates. Meaning, the furthest "left" or "top" value can be 0.
        """
        return self._location

    @property
    def mask_on_image(self):
        """
        Notes: the mask_on_image array (Nx x Ny) is in array coordinates (i.e. Nx IS the columns of the array).
        So, if you wanted to plot the mask_on_image array using imshow(), you would need to modify the location_on_template
        coordinates in order to get the location coordinates and mask coordinates into the same coordinate system.
        """
        return self._mask_on_image

    @property
    def image(self):
        return self._image

    @property
    def template(self):
        return self._template

    @property
    def location_on_template(self):
        """
        Important Notes:
            * the location_on_template tuple (x, y) is in plotting coordinates (so you can scatter plot).
            * plotting coordinates means if the x-location on template was 10, then the x-location on the template
            array would be the 11th index of the columns.
            * b/c the template should always have an odd-numbered side length, this means the location on template
            should always be an odd number.

            For example:
                template x-shape = 47
                template x-indices = 0 : 46
                x-location template = 23
        """
        return self._location_on_template

    @property
    def mask_on_template(self):
        return self._mask_on_template

    @property
    def template_contour(self):
        return self._template_contour

    @property
    def cm(self):
        return self._cm

    @property
    def max_sim(self):
        return self._max_sim

    @property
    def z(self):
        return self._z