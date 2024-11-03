# IdptSetup.py


import numpy as np


class IdptSetup(object):

    def __init__(self, inputs, outputs, processing, z_assessment, optics=None):
        """

        :param inputs:
        :param outputs:
        :param processing:
        :param z_assessment:
        :param optics:
        """
        self.inputs = inputs
        self.outputs = outputs
        self.processing = processing
        self.z_assessment = z_assessment
        self.optics = optics


class inputs(object):
    def __init__(self, dataset, image_collection_type, image_path, image_file_type, image_base_string,
                 calibration_z_step_size, baseline_image, hard_baseline=True, image_subset=None):
        """

        :param dataset:
        :param image_collection_type:
        :param image_path:
        :param image_file_type:
        :param image_base_string:
        :param calibration_z_step_size:
        :param baseline_image:
        :param hard_baseline:
        :param image_subset:
        """
        # image collection ID and type
        self.dataset = dataset
        self.image_collection_type = image_collection_type

        # file paths
        self.image_path = image_path
        self.image_file_type = image_file_type

        # file information
        self.image_base_string = image_base_string

        # dataset
        self.image_subset = image_subset
        self.calibration_z_step_size = calibration_z_step_size
        self.baseline_image = baseline_image
        self.hard_baseline = hard_baseline


class outputs(object):
    def __init__(self, results_path, save_id_string, save_plots):
        """

        :param results_path:
        :param save_id_string:
        :param save_plots:
        """
        self.results_path = results_path
        self.save_plots = save_plots
        self.save_id_string = save_id_string


class processing(object):
    def __init__(self, min_particle_area, max_particle_area, template_padding,
                 same_id_threshold, stacks_use_raw, cropping, background_subtraction,
                 preprocessing, thresholding, xy_displacement=None):
        """

        :param min_particle_area:
        :param max_particle_area:
        :param template_padding:
        :param same_id_threshold:
        :param stacks_use_raw:
        :param cropping:
        :param background_subtraction:
        :param preprocessing:
        :param thresholding:
        :param xy_displacement:
        """

        self.template_padding = template_padding
        self.min_particle_area = min_particle_area
        self.max_particle_area = max_particle_area
        self.same_id_threshold = same_id_threshold
        self.stacks_use_raw = stacks_use_raw
        self.background_subtraction = background_subtraction
        self.cropping = cropping
        self.preprocessing = preprocessing
        self.thresholding = thresholding

        if xy_displacement is None:
            xy_displacement = [[0, 0]]
        self.xy_displacement = xy_displacement


class z_assessment(object):
    def __init__(self, infer_method, use_stack_id=None):
        """

        :param infer_method:
        :param use_stack_id:
        """
        self.infer_method = infer_method
        self.use_stack_id = use_stack_id


class optics(object):
    def __init__(self, particle_diameter=None, demag=None, magnification=None, numerical_aperture=None,
                 focal_length=None,
                 ref_index_medium=None, ref_index_lens=None, pixel_size=None,
                 pixel_dim_x=None, pixel_dim_y=None, bkg_mean=None, bkg_noise=None, points_per_pixel=None, n_rays=None,
                 gain=None, cyl_focal_length=None, wavelength=None, z_range=None):
        """

        :param particle_diameter:
        :param demag:
        :param magnification:
        :param numerical_aperture:
        :param focal_length:
        :param ref_index_medium:
        :param ref_index_lens:
        :param pixel_size:
        :param pixel_dim_x:
        :param pixel_dim_y:
        :param bkg_mean:
        :param bkg_noise:
        :param points_per_pixel:
        :param n_rays:
        :param gain:
        :param cyl_focal_length:
        :param wavelength:
        :param z_range:
        """
        self.particle_diameter = particle_diameter
        self.demag = demag
        self.magnification = magnification
        self.numerical_aperture = numerical_aperture
        self.focal_length = focal_length
        self.ref_index_medium = ref_index_medium
        self.ref_index_lens = ref_index_lens
        self.pixel_size = pixel_size
        self.pixel_dim_x = pixel_dim_x
        self.pixel_dim_y = pixel_dim_y
        self.bkg_mean = bkg_mean
        self.bkg_noise = bkg_noise
        self.points_per_pixel = points_per_pixel
        self.n_rays = n_rays
        self.gain = gain
        self.cyl_focal_length = cyl_focal_length
        self.wavelength = wavelength
        self.z_range = z_range

        # effective magnification
        self.effective_magnification = demag * magnification

        # pixels per particle
        self.pixels_per_particle_in_focus = particle_diameter * self.effective_magnification / pixel_size

        # field of view
        self.field_of_view = pixel_size * pixel_dim_x / self.effective_magnification

        # microns per pixel scaling factor
        self.microns_per_pixel = pixel_size / self.effective_magnification

        # Rayleigh Criterion (maximum lateral resolution)
        self.Rayleigh_criterion = 0.61 * wavelength / numerical_aperture

        # depth of field
        self.depth_of_field = ref_index_medium * wavelength / numerical_aperture ** 2 + ref_index_medium * pixel_size / \
                              (self.effective_magnification * numerical_aperture)

        # constants for stigmatic/astigmatic imaging systems (ref: Rossi & Kahler 2014, DOI 10.1007/s00348-014-1809-2)
        self.c1 = 2 * (ref_index_medium ** 2 / numerical_aperture ** 2 - 1) ** -0.5
        self.c2 = (particle_diameter ** 2 + 1.49 * wavelength ** 2 * (
                    ref_index_medium ** 2 / numerical_aperture ** 2 - 1)) ** 0.5

        # create a measurement depth
        z_space = np.linspace(start=-z_range, stop=z_range, num=250)

        # particle image diameter with distance from focal plane (stigmatic system)
        # (ref 1: Rossi & Kahler 2014, DOI 10.1007 / s00348-014-1809-2)
        self.particle_diameter_z1 = self.effective_magnification * (self.c1 ** 2 * z_space ** 2 + self.c2 ** 2) ** 0.5
        # (ref 2: Rossi & Kahler 2014, )
        self.particle_diameter_z2 = self.effective_magnification * self.particle_diameter

    def stigmatic_diameter_z(self, z_space, z_zero):
        """

        :param z_space:
        :param z_zero:
        :return:
        """
        # if units of z are not in microns, adjust prior to calculating the intensity profile
        mod = False
        if np.max(np.abs(z_space)) > 1:
            z_space = z_space * 1e-6
            mod = True

        # create dense z-space for smooth plotting
        z_space = np.linspace(start=np.min(z_space), stop=np.max(z_space), num=250)

        # particle image diameter with distance from focal plane (stigmatic system)
        # (ref 1: Rossi & Kahler 2014, DOI 10.1007 / s00348-014-1809-2)
        particle_diameter_z1 = self.effective_magnification * (
                    self.c1 ** 2 * (z_space - z_zero * 1e-6) ** 2 + self.c2 ** 2) ** 0.5

        # convert units to microns
        particle_diameter_z1 = particle_diameter_z1 * 1e6

        if mod is True:
            z_space = z_space * 1e6

        return z_space, particle_diameter_z1

    def stigmatic_maximum_intensity_z(self, z_space, max_intensity_in_focus, z_zero=0, background_intensity=0,
                                      num=250):
        """
        maximum intensity with distance from the focal plane (stigmatic system)

        z_space: list or array like; containing all the non-normalized z-coordinates in the collection
        max_intensity_in_focus: float; maximum mean particle intensity in the collection
            * Note: this is the maximum of the mean of the contour areas (region of signal) in the collection.
        """
        # if units of z are not in microns, adjust prior to calculating the intensity profile
        mod = False
        if np.max(np.abs(z_space)) > 1:
            z_space = z_space * 1e-6
            mod = True

        # create dense z-space for smooth plotting
        z_space = np.linspace(start=np.min(z_space), stop=np.max(z_space), num=num)

        # calculate the intensity profile as a function of z
        stigmatic_intensity_profile = self.c2 ** 2 / ((self.c1 ** 2 * (z_space - z_zero * 1e-6) ** 2 + self.c2 ** 2) **
                                                      0.5 * (self.c1 ** 2 * (z_space - z_zero * 1e-6) ** 2 +
                                                             self.c2 ** 2) ** 0.5)

        # normalize the intensity profile so it's maximum value is equal to the max_intensity_in_focus
        stigmatic_intensity_profile = np.round(
            (max_intensity_in_focus - background_intensity) * stigmatic_intensity_profile / \
            np.max(stigmatic_intensity_profile) + background_intensity,
            1)

        if mod is True:
            z_space = z_space * 1e6

        return z_space, stigmatic_intensity_profile

    # maximum intensity with distance from the focal plane (astigmatic system)
    # (ref: Rossi & Kahler 2014, DOI 10.1007 / s00348-014-1809-2)