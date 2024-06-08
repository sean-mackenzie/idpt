# utils.utils.particle_identification.py

import numpy as np
from skimage.measure import label, regionprops, find_contours

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def apply_threshold(img, parameter, show_threshold):
    method = list(parameter.keys())[0]

    if method == 'none':
        thresh_val = 0
    elif method == 'manual':
        args = parameter[method]
        if not isinstance(parameter[method], list):
            thresh_val = parameter[method]
        elif len(parameter[method]) == 1:
            thresh_val = parameter[method]
        else:
            raise ValueError("Manual threshold not understood.")
    else:
        raise ValueError("method must be one of ['none', 'manual']")
    thresh_img = img > thresh_val

    if show_threshold:
        num_signal_pixels = np.count_nonzero(thresh_img)
        num_pixels = thresh_img.size
        image_density = num_signal_pixels / num_pixels

        fig, ax = plt.subplots(ncols=2, sharey=True)
        ax[0].imshow(img)
        ax[0].set_title(r'$I_{o}(\overline{I} \pm \sigma_{I}=$' +
                        '({}, {} +/- {}, {})'.format(np.min(img),
                                                     np.round(np.mean(img), 1),
                                                     np.round(np.std(img), 2),
                                                     np.max(img),
                                                     ),
                        fontsize=8,
                        )
        ax[0].set_yticks([0, img.shape[1]])
        ax[0].set_xticks([0, img.shape[1]])

        ax[1].imshow(thresh_img)
        ax[1].set_title(r'$N_{s} = $' + '{}'.format(np.round(image_density, 4)),
                        fontsize=8,
                        )
        ax[1].set_xticks([0, thresh_img.shape[1]])

        plt.tight_layout()
        plt.show()
        plt.close()

    return thresh_img


def identify_contours(particle_mask, intensity_image):

    # label the particle mask without segmentation
    label_image = label(particle_mask)

    # region properties
    regions = regionprops(label_image=label_image, intensity_image=intensity_image)

    # check if any regions are within the same_id_threshold and remove regions if max/mean intensity is much lower
    labels = []
    weighted_centroids = []
    max_intensities = []
    mean_intensities = []
    for region in regions:
        labels.append(region.label)
        weighted_centroids.append(region.weighted_centroid)
        max_intensities.append(region.max_intensity)
        mean_intensities.append(region.mean_intensity)

    labels_to_remove = []
    contour_coords = []
    new_regions = []
    for region in regions:
        if region.label not in labels_to_remove:
            new_regions.append(region)

            zero_array = np.zeros_like(intensity_image.T, dtype=int)
            points = region.coords
            zero_array[points[:, 1], points[:, 0]] = 1

            cont = find_contours(zero_array)

            if len(cont) == 0:
                continue
            else:
                contour = cont[0].astype(int)
                contour_coords.append(contour)

    return label_image, new_regions, contour_coords


def pad_and_center_region(cX, cY, bbox, padding):
    # make the bounding box a square (w = h)
    if bbox[2] > bbox[3]:
        bbox = (bbox[0], bbox[1], bbox[2], bbox[2])
    if bbox[3] > bbox[2]:
        bbox = (bbox[0], bbox[1], bbox[3], bbox[3])

    # make the bounding box dimensions odd (w, h == odd number) to center the particle image on center pixel
    assert bbox[2] == bbox[3]
    if bbox[2] % 2 == 0:
        bbox = (bbox[0], bbox[1], bbox[2] + 1, bbox[3] + 1)

    # pad the bounding box to ensure the entire particle is captured
    bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding * 2, bbox[3] + padding * 2]

    # center bounding box on computed contour center
    bbox = [cX - int(np.floor(bbox[2] / 2)), cY - int(np.floor(bbox[3] / 2)), bbox[2], bbox[3]]

    return bbox, (cX, cY)