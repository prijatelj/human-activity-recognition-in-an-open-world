"""Classes that augment the given image in different ways."""
from abc import abstractmethod
from collections import OrderedDict
import glob
import inspect
import logging
from math import floor
import os
import sys

import cv2
import numpy as np
from scipy.interpolate import griddata
from torch.utils.data import Dataset # TODO use IterableDataset

from exputils.ml.generic_predictors import Stateful


# TODO make torch DataLoader versions of these that can be chained together.
#   Perhaps, make a generic class that allows the user to give the function
#   `augment()`. And make another class that combines torch.DataLoader with
#   these calsses so that they may be used outside of torch if desired.

# TODO handle slices eventually in these __getitems__()

# TODO consider functional programming approach so `iterable` is not always a
# necessary function to pass in, and then the made class is simply a function
# that applies to any iterable.

class StatefulIterable(Stateful):
    """A stateful iterable object that includes base checks on the iterable."""
    def __init__(self, iterable=None):
        """Iterates through the given iterable, applying the augmentation."""
        try:
            iter(iterable)
        except TypeError:
            if iterable is not None:
                raise TypeError(' '.join([
                    '`iterable` must be an iterable object or None,',
                    f'not {type(iterable)}',
                ]))
        self.iterable = iterable

    def __len__(self):
        """Iterates through the given iterable, applying the augmentation."""
        if self.iterable is None:
            raise TypeError('`iterable` is not set! Cannot get length.')
        return len(self.iterable)

    @abstractmethod
    def __getitem__(self, idx):
        """Does not return anything. Only performs the typical checks."""
        if self.iterable is None:
            raise TypeError('`iterable` is not set! Cannot get item.')
        if idx >= len(self):
            raise IndexError(f'Index `{idx}` out of range `{len(self)}`.')

    def set_iter(self, iterable):
        """Iterates through the given iterable, applying the augmentation."""
        try:
            iter(iterable)
        except TypeError:
            raise TypeError(
                f'`iterable` must be an iterable object, not {type(iterable)}'
            )
        self.iterable = iterable


class Augmenter(StatefulIterable, Dataset):
    """The base augmenter class that includes checks and basic agumentation."""
    def __getitem__(self, idx):
        """Iterates through the given iterable, applying the augmentation."""
        super(Augmenter, self).__getitem__(idx)

        item = self.iterable[idx]
        # TODO be wary that changing the HWRItem's image may result in issues
        # down the line, and so copying would be necessary! Check this (w/ a
        # script cuz i belieave i did before and it was ok, so i need
        # essentially a unit test... It worked before being made a Torch
        # Dataset)! This is probably fine because everytime getitem is called
        # it loads the image from disk, no buffer. Only issue would be in
        # PyTorch if it tried to be efficient.
        item.image = self.augment(item.image)
        if hasattr(item, 'labels'):
            item.labels = item.labels.copy()
            # TODO extend such that it is a list that grows in size, keeping
            # the order of augmentations.
            item.labels['appearance'] = type(self).__name__

        return item

    @abstractmethod
    def augment(self, image):
        raise NotImplementedError()


class StochasticAugmenter(Augmenter):
    """An augmenter that uses a stochastic augmentation method thus needing its
    random state and number of times the augmentation is applied to each item
    in the original iterable.
    """
    def __init__(
        self,
        augs_per_item,
        include_original=False,
        rng=None,
        iterable=None,
    ):
        super(StochasticAugmenter, self).__init__(iterable)

        self.augs_per_item = augs_per_item
        self.include_original = include_original

        if rng is None or isinstance(rng, int):
            self.rng = np.random.Generator(np.random.PCG64(rng))
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            raise TypeError(' '.join([
                '`rng` expected to be of type None, int, or',
                'np.random.Generator, but instead recieved argument of',
                f'type: {type(rng)}',
            ]))

    def __len__(self):
        # This assumes you increase the size when include_original = True
        return super(StochasticAugmenter, self).__len__() * (
            self.augs_per_item + 1 if self.include_original
            else self.augs_per_item
        )

    def __getitem__(self, idx):
        """Iterates through the original iterable of images and extends that
        iterable to be of length: `len(iterable) * augs_per_item` or
        `augs_per_item + 1` if `include_original` is True. If
        `include_original` is True, then the first `len(iterable)` items are
        the original items, unaugmented.
        """
        StatefulIterable.__getitem__(self, idx)

        if self.include_original and idx < len(self.iterable):
            # This ensures the first pass is unaugmented
            return super(StochasticAugmenter, self).__getitem__(idx)

        # The following modulo wraps this back around if greater.
        return super(StochasticAugmenter, self).__getitem__(
            idx % len(self.iterable)
        )

    # TODO add label iterator so labels are easily obtainable w/o loading or
    # processing the image


class ElasticTransform(StochasticAugmenter):
    """Performs the elastic transform on the given images via grid distortion.

    Attributes
    ----------
    augs_per_item : int
    include_original : bool
    mesh_interval : int, tuple(int, int)
        Results in a tuple of the width and height intervals `(height_interval,
        width_interval)`. If given an int, then the width and height intervals
        are both set to the given int.
    mesh_std : float, tuple(float, float)
        Results in a tuple of the width and height standard deviations
        `(height_std, width_std)`. If given a float, then the height and width
        standard deviations are both set to the given float.
    interpolation : str
        Method of interpolation, either linear or cubic. Uses cv2.INTER_LINEAR
        or cv2.INTER_CUBIC respectively.
    interpolation_cv2 : cv2.INTER_LINEAR, cv2.INTER_CUBIC
        The opencv interpolation to apply to the images.
    fit_interval_to_image : bool
    draw_grid_lines : bool
    rng : None, int, np.random.Generator
    """
    def __init__(
        self,
        mesh_interval=25,
        mesh_std=3.0,
        interpolation='linear',
        fit_interval_to_image=True,
        draw_grid_lines=False,
        *args,
        **kwargs,
    ):
        super(ElasticTransform, self).__init__(*args, **kwargs)

        if (
            isinstance(mesh_interval, tuple)
            and len(mesh_interval) == 2
            and isinstance(mesh_interval[0], int)
            and isinstance(mesh_interval[1], int)
        ):
            self.mesh_interval = mesh_interval
        elif isinstance(mesh_interval, int):
            self.mesh_interval = (mesh_interval, mesh_interval)
        else:
            raise TypeError(' '.join([
                '`mesh_interval` expected type int or tuple(int, int), not',
                f'{type(mesh_interval)}',
            ]))

        if (
            isinstance(mesh_std, tuple)
            and len(mesh_std) == 2
            and isinstance(mesh_std[0], float)
            and isinstance(mesh_std[1], float)
        ):
            self.mesh_std = mesh_std
        elif isinstance(mesh_std, float):
            self.mesh_std = (mesh_std, mesh_std)
        else:
            raise TypeError(' '.join([
                '`mesh_std` expected type float or tuple(float, float), not',
                f'{type(mesh_std)}',
            ]))

        if interpolation == 'linear':
            self.interpolation_cv2 = cv2.INTER_LINEAR
        elif interpolation == 'cubic':
            self.interpolation_cv2 = cv2.INTER_CUBIC
        else:
            raise ValueError(' '.join([
                '`interpolation` expected "linear" or "cubic", not',
                f'{interpolation}.',
            ]))
        self.interpolation = interpolation

        self.fit_interval_to_image = fit_interval_to_image
        self.draw_grid_lines = draw_grid_lines

    def augment(self, image):
        height, width = image.shape[:2]

        if self.fit_interval_to_image:
            # Change interval so it fits the image size
            h_ratio = max(1, round(height / float(self.mesh_interval[0])))
            w_ratio = max(1, round(width / float(self.mesh_interval[1])))

            mesh_interval = (height / h_ratio, width / w_ratio)
        else:
            mesh_interval = self.mesh_interval

        # Get control points
        source = np.mgrid[
            0:height + mesh_interval[0]:mesh_interval[0],
            0:width + mesh_interval[1]:mesh_interval[1]
        ]
        source = source.transpose(1, 2, 0).reshape(-1, 2)

        if self.draw_grid_lines:
            if len(image.shape) == 2:
                color = 0
            else:
                color = np.array([0, 0, 255])
            for src in source:
                image[int(src[0]):int(src[0]) + 1, :] = color
                image[:, int(src[1]):int(src[1]) + 1] = color

        # Perturb source control points
        destination = source.copy()
        source_shape = source.shape[:1]
        destination[:, 0] = destination[:, 0] + self.rng.normal(
            0.0,
            self.mesh_std[0],
            size=source_shape,
        )
        destination[:, 1] = destination[:, 1] + self.rng.normal(
            0.0,
            self.mesh_std[1],
            size=source_shape,
        )

        # Warp image
        grid_x, grid_y = np.mgrid[0:height, 0:width]
        grid_z = griddata(
            destination,
            source,
            (grid_x, grid_y),
            method=self.interpolation,
        ).astype(np.float32)

        map_x = grid_z[:, :, 1]
        map_y = grid_z[:, :, 0]

        return cv2.remap(
            image,
            map_x,
            map_y,
            self.interpolation_cv2,
            borderValue=(255, 255, 255),
        )


class Noise(StochasticAugmenter):
    """Add Gaussian noise to the image."""
    def __init__(self, mean=0, std=10, *args, **kwargs):
        super(Noise, self).__init__(*args, **kwargs)

        # TODO figure out how this object can carry a cv2 RNG object...
        #self.cv2_rng = cv2.RNG_NORMAL(self.rng.bit_generator.state['uinteger'])

        self.mean = mean
        self.std = std

    def augment(self, image):
        if len(image.shape) != 3:
            raise ValueError(
                f'`image` shape expected to be 3 dims BGR, not {image.shape}'
            )
        if image.dtype != 'uint8':
            raise ValueError(
                f'`image` dtype expected to be uint8. BGR. not {image.dtype}'
            )
        noise = np.zeros(image.shape[:2])
        cv2.randn(noise, self.mean, self.std)
        noise = np.stack([noise]*3, axis=-1)
        result = (image / 255) + noise
        # Ensure the range [0, 255]
        result = (np.minimum(np.maximum(result, 0), 1) * 255).astype('uint8')
        return image + result


class Blur(StochasticAugmenter):
    """Gaussian blur the image."""
    def __init__(self, ksize, sigmaX, sigmaY=0, *args, **kwargs):
        super(Blur, self).__init__(*args, **kwargs)

        self.ksize = tuple(ksize)
        self.sigmaX = sigmaX
        self.sigmaY = sigmaY

    def augment(self, image):
        return cv2.GaussianBlur(image, self.ksize, self.sigmaX, self.sigmaY)


class InvertColor(Augmenter):
    def __init__(self, iterable=None):
        super(InvertColor, self).__init__(iterable)

    def augment(self, image):
        if len(image.shape) != 3:
            raise ValueError(
                f'`image` shape expected to be 3 dims BGR, not {image.shape}'
            )
        if image.dtype != 'uint8':
            raise ValueError(
                f'`image` dtype expected to be uint8. BGR. not {image.dtype}'
            )
        return 255 - image


class Reflect(Augmenter):
    """Generalized reflection class that reflects a 2D matrix."""
    def __init__(self, axis, iterable=None):
        super(Reflect, self).__init__(iterable)
        self.axis = axis

    def augment(self, image):
        # TODO make optionally torch flip, based input
        return np.flip(image, axis=self.axis)


class Reflect_X(Reflect):
    def __init__(self, iterable=None):
        super(Reflect_X, self).__init__(0, iterable)


class Reflect_Y(Reflect):
    def __init__(self, iterable=None):
        super(Reflect_Y, self).__init__(1, iterable)


class Antique(StochasticAugmenter):
    """Given the image, assumed to be grayscale/binary with white background,
    the text line image is blended with a random selection from a set of
    background paper images. A slice from the random selected background image
    that fits the line image is applied to the line image. This is to simulate
    text on antique papers.

    Attributes
    ----------
    background_images : [np.ndarray]
        List of background images to use for replacing the background of text
        images.
    """
    def __init__(self, backgrounds_dir, grayscale=True, *args, **kwargs):
        super(Antique, self).__init__(*args, **kwargs)

        # Load the images from the backgrounds directory
        self.background_images = []
        for img_path in glob.iglob(os.path.join(backgrounds_dir, '*')):
            if grayscale:
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # We by default expect BGR for all other images, thus repeat
                image = np.repeat(np.expand_dims(image, 2), 3, 2)
            else:
                image = cv2.imread(img_path)

            if image is None:
                raise IOError(
                    f'Read image is None. Invalid filepath: {img_path}',
                )

            self.background_images.append(image)

        if not self.background_images:
            raise ValueError('No images were loaded!')

    def augment(
        self,
        image,
        bg_image_id=None,
        approach='minimum',
        color=[255, 255, 255]
    ):
        # Set offsets to random part of background
        if bg_image_id is None:
            bg_image = self.background_images[
                self.rng.integers(len(self.background_images))
            ]
        else:
            bg_image = self.background_images[bg_image_id]

        if bg_image.shape[1] >= image.shape[1]:
            x_offset = int(
                self.rng.random()
                * (bg_image.shape[1] - image.shape[1])
            )
        else:
            x_offset = 0

        if bg_image.shape[0] >= image.shape[0]:
            y_offset = int(
                self.rng.random()
                * (bg_image.shape[0] - image.shape[0])
            )
        else:
            y_offset = 0

        bg_image_select = bg_image[
            y_offset:y_offset + image.shape[0],
            x_offset:x_offset + image.shape[1],
            :,
        ]

        if image.shape[0:2] != bg_image.shape[0:2]:
            bg_image_select = cv2.resize(
                bg_image_select,
                (image.shape[1], image.shape[0]),
            )
            blended = np.copy(image)

        for i in range(image.shape[2]):
            if approach == "minimum":
                blended[:, :, i] = np.minimum(
                    image[:, :, i],
                    bg_image_select[:, :, i],
                )
            elif approach == "factor":
                img = np.copy(image[:, :, i])

                # Darker is letters
                factor = (255 - img) / 255.0
                blended[:, :, i] = bg_image_select[:, :, i] \
                    * (1 - factor) + img * factor
            else:
                img = np.copy(image[:, :, i])
                blended[:, :, i] = bg_image_select[:, :, i]
                blended[:, :, i][img < 250] = color[i] \
                    * ((255 - img) / 255.0).astype(np.uint8)[img < 250]

        return blended


class SplitAugmenters(Augmenter):
    """Given multiple augmenters, performs them equally on the iterable's
    items.  This does not expand the given iterable, but rather modifies
    different items within it as balanced as possible given the augmenters.
    This is accomplished via modulo.

    Attributes
    ----------
    augmenters : [Augmenter]
        A list of Augmenter instances to be applied to the original iterable in
        order.
    include_original : bool = False
        If True, includes the original iterable in the contents of the
        reuslting iterable object. Otherwise, the resulting iterable only
        includes augmented items.
    iterable : Iterable
        The iterable object whose content is equally augmented to provide
        a near equal distribution of augmentations from each augmenter.
    rng : None | int | np.random.Generator = None
        The random seed or generator to use for all StochasticAugmentors.
    """
    def __init__(
        self,
        augmenters,
        include_original=False,
        iterable=None,
        rng=None,
        #weight=None,
    ):
        super(SplitAugmenters, self).__init__(iterable)

        self.include_original = include_original
        self.rng = rng
        augmenters = OrderedDict(augmenters)

        # TODO allow the user to specify a weighting / distribution of the
        # augmentations and modify the resulting augmented iterable accordinly.

        # Obtain all previously initialized augmenter classes. This is a
        # meta-programming shorthand to avoid writing out the list above.
        class_members = {name: value for name, value in inspect.getmembers(
            sys.modules[__name__],
            lambda member: (
                inspect.isclass(member)
                and member.__module__ == __name__
                and issubclass(member, Augmenter)
            ),
        )}

        # Loop through the augmenters and create them using the list of
        # parameters, if not already provided as an Augmenter instance.
        for key, args in augmenters.items():
            if issubclass(type(args), Augmenter):
                continue
            if 'Reflect' in key and key not in {'Reflect_X', 'Reflect_Y'}:
                # Allow Reflect_0 and Reflect_1 to create Reflect Augs
                if augmenters[key]['axis'] == 0:
                    augmenters[key] = class_members['Reflect'](**augmenters[key])
                elif augmenters[key]['axis'] == 1:
                    augmenters[key] = class_members['Reflect'](**augmenters[key])
            elif key == 'InvertColor':
                # InvertColor has no args, handling separately
                    augmenters[key] = class_members[key]()
            elif key not in class_members:
                raise KeyError(
                    f'Given key for an augmenter that does not exist: {key}'
                )
            else:
                if issubclass(class_members[key], StochasticAugmenter):
                    augmenters[key] = class_members[key](
                        rng=rng,
                        **augmenters[key],
                    )
                else:
                    augmenters[key] = class_members[key](**augmenters[key])

        self.augmenters = list(augmenters.values())

    def __getitem__(self, idx):
        if self.include_original:
            # Do not apply any augmenter on every len(augmenters) item
            aug_idx = idx % (len(self.augmenters) + 1)

            if aug_idx == len(self.augmenters):
                item = self.iterable[idx]
                if hasattr(item, 'labels'):
                    item.labels = item.labels.copy()
                    item.labels['appearance'] = 'no_aug'
                return item
        else:
            # No unaugmented items
            aug_idx = idx % len(self.augmenters)

        augmenter = self.augmenters[aug_idx]

        item = self.iterable[idx]
        item.image = augmenter.augment(item.image)

        if hasattr(item, 'labels'):
            name = type(augmenter).__name__
            if name == 'Reflect':
                name = f'{name}_{augmenter.axis}'
            item.labels = item.labels.copy()
            item.labels['appearance'] = name
        return item

    def augment(self, image, aug):
        return aug.augment(image)


class ParallelAugmenters(SplitAugmenters):
    """Given multiple augmenters, duplicates the original iterable such that
    each augmenter is executed at least once per sample or as many times as the
    specific augmenter repeats per sample.  This expands the given iterable.

    Attributes
    ----------
    augmenters : [Augmenter]
        An ordered dictionary of keys as string augmenter identifiers mapping
        to their respective Augmenter.
    include_original : bool = True
        If True, includes the original iterable in the contents of the
        reuslting iterable object. Otherwise, the resulting iterable only
        includes augmented items.
    iterable : Iterable
        An iterable object whose content is duplicated as necessary to provide
        an augmentation from each augmenter per sample.
    rng : None | int | np.random.Generator = None
        The random seed or generator to use for all StochasticAugmentors.
    """
    # TODO consider actually parallelizing this process. lol
    def __init__(
        self,
        augmenters,
        include_original=False,
        iterable=None,
        rng=None,
        #weight=None,
    ):
        super(ParallelAugmenters, self).__init__(
            augmenters,
            include_original,
            iterable,
            rng,
        )
        # TODO multiply the size of the iterable. Giving same text and writer
        # to different backgrounds.

        # TODO allow the user to specify a weighting / distribution of the
        # augmentations and modify the resulting augmented iterable accordinly.

        #if not include_original:
        #    raise NotImplementedError('Currently does not work. Needs fixed!')
        self.__calc_num_augs()

    def __calc_num_augs(self):
        """Computes and sets the number of augmentations applied to each image.
        This is not the the number of augmenters in this class, but the number
        of individual augmentations applied to a single image, accounting for
        an augmenter with multiple augmentations in itself, e.g.
        StochasticAugmenters."""
        augs_per_aug = []
        for aug in self.augmenters:
            if isinstance(aug, StochasticAugmenter):
                augs_per_aug.append(
                    aug.augs_per_item + 1 if aug.include_original \
                    else aug.augs_per_item
                )
            else:
                augs_per_aug.append(1)

        augs_per_aug = np.array(augs_per_aug)
        self._augs_cumsum = augs_per_aug.cumsum()
        if self.include_original:
            self._num_augs = augs_per_aug.sum() + 1
        self._num_augs = augs_per_aug.sum()

    def __len__(self):
        """Iterates through all augmenters obtaining their length and summing"""
        return self._num_augs * super(ParallelAugmenters, self).__len__()

    def __getitem__(self, idx):
        StatefulIterable.__getitem__(self, idx)

        dup_idx = floor(idx / self._num_augs)

        # Map to approriate aug idx when an aug has multiple augs itself
        aug_idx = idx % self._num_augs
        if len(self.augmenters) < (
            self._num_augs - 1 if self.include_original else self._num_augs
        ):
            # At least one augmenter has multiple augmentations
            # TODO this could be more efficient.
            for i, cumsum in enumerate(self._augs_cumsum):
                if aug_idx < cumsum:
                    aug_idx = i
                    break

        if self.include_original and aug_idx == len(self.augmenters):
            # The iterable's item that is being augmented, handling duplicaiton
            item = self.iterable[dup_idx]
            if hasattr(item, 'labels'):
                item.labels = item.labels.copy()
                item.labels['appearance'] = 'no_aug'
            return item

        augmenter = self.augmenters[aug_idx]

        item = self.iterable[dup_idx]
        item.image = augmenter.augment(item.image)

        if hasattr(item, 'labels'):
            name = type(augmenter).__name__
            if name == 'Reflect':
                name = f'{name}_{augmenter.axis}'
            item.labels = item.labels.copy()
            item.labels['appearance'] = name
        return item

    def augment(self, image, aug):
        return aug.augment(image)


class SequentialAugmenters(Augmenter):
    """Applies a series of augmentations in order to each iterable's item.

    Attributes
    ----------
    augmenters : [Augmenter]
        A list of keys as string augmenter identifiers mapping to their
        respective Augmenter.
    include_original : bool = False
        If True, includes the original iterable in the contents of the
        reuslting iterable object. Otherwise, the resulting iterable only
        includes augmented items.
    iterable : Iterable
        An iterable object whose items are augmented using the ordered series
        of augmentations.
    """
    def __init__(self):
        raise NotImplementedError()

# TODO Sequential Augmentations manager. Split and Dup handle the
# paralelization, and chaining them handles sequential (easily chained via
# passing the iterables and a funtional approach incoming)

# TODO be able to generate the appearance column in labels dataframe for post
# analysis. Currently involves iterating over the iterable again and then
# saving the appearance results, which is inefficient when not actually using
# the image that is loaded and preprocessed/augmented.

# TODO EffectMap / EffectMask: make it so the above effects only apply to parts
# of the image given some distribution of effect. Binary for on/off, or
# gradient of effect where applicable. e.g. partial noise, partial blur.
# Essentially a mask for the effects.

# Namespace shortcuts for convenience.
#SplitAug = SplitAugmenters
#ParallelAug = ParallelAugmenters
