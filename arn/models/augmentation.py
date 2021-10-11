"""Classes that augment the given image in different ways."""
from abc import abstractmethod
from collections import OrderedDict
import glob
import inspect
import logging
from math import floor
import os
import sys
import torchvision.transforms as T
import torchvision.transforms.functional as F
import cv2
import numpy as np
from scipy.interpolate import griddata
from torch.utils.data import Dataset # TODO use IterableDataset
import torch
from torchvision.transforms.functional import InterpolationMode

from exputils.ml.generic_predictors import Stateful

# TODO tell derek about breaking change adding param output

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
        augs_per_item=1,
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


class ColorJitter(StochasticAugmenter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, *args, **kwargs):
        super(ColorJitter, self).__init__(*args, **kwargs)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.cj = T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        #TODO figure out if we care about the random state or it is fine to just record everything


    def augment(self, image):
        # TODO these are probably bad for the purposes of this, since pytorch expects it to be C,H,W instead of H,W,C
        if len(image.shape) != 3:
            raise ValueError(
                f'`image` shape expected to be 3 dims BGR, not {image.shape}'
            )
        if image.dtype != 'uint8':
            raise ValueError(
                f'`image` dtype expected to be uint8. BGR. not {image.dtype}'
            )
        image = torch.tensor(image).permute(2, 0, 1)
        # to keep the parameters for the transform, i had to copy it from the forward method Hope it doesn't change :shrug:
        params = self.cj.get_params(self.cj.brightness, self.cj.contrast, self.cj.saturation, self.cj.hue)

        return self.aug(image,params).permute(1,2,0).numpy(),params
    @staticmethod
    def aug(img, params):
        # image in pytorch format
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = params
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)
        return img


class Noise(StochasticAugmenter):
    """Add Gaussian noise to the image."""
    def __init__(self, mean=0, std=10,*args, **kwargs):
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
        return image + result, result

    def aug(self, image, params):
        return image + params

class Blur(StochasticAugmenter):
    """Gaussian blur the image."""
    def __init__(self, ksize, sigma_min=0.1, sigma_max=0.2, *args, **kwargs):
        super(Blur, self).__init__(*args, **kwargs)

        self.ksize = tuple(ksize)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def augment(self, image):
        image = torch.tensor(image).permute(2, 0, 1)
        sigma = T.GaussianBlur.get_params(self.sigma_min, self.sigma_max)
        params = self.ksize, [sigma, sigma]
        return self.aug(image,params).permute(1,2,0).numpy(),params
    @staticmethod
    def aug(img,params):
        ksize, sigma = params
        return F.gaussian_blur(img, ksize, sigma)


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
        image = torch.tensor(image).permute(2, 0, 1)
        return F.invert(image).permute(1,2,0).numpy(), True
    @staticmethod
    def aug(image,params):
        image = torch.tensor(image).permute(2, 0, 1)
        if params:
            return F.invert(image)
        else:
            return image


class PerspectiveTransform(StochasticAugmenter):
    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=0, *args, **kwargs):
        super(PerspectiveTransform, self).__init__(*args, **kwargs)
        if fill == -1:
            fill = list((255*torch.rand(3)).numpy().astype(np.uint8))
        self.pt  = T.RandomPerspective(distortion_scale=distortion_scale, p=p, interpolation=interpolation, fill=fill)

    def augment(self, image):
        image = torch.tensor(image).permute(2, 0, 1)
        width, height = F._get_image_size(image)
        params = T.RandomPerspective.get_params(width, height, distortion_scale=self.pt.distortion_scale)
        params += (self.pt.p,self.pt.interpolation,self.pt.fill)
        return self.aug(image, params).permute(1,2,0).numpy(), params


    @staticmethod
    def aug(img,params):
        # width, height = F._get_image_size(img)
        # startpoints, endpoints = self.get_params(width, height, self.pt.distortion_scale)
        # F.perspective(img, startpoints, endpoints, self.interpolation, fill)
        return F.perspective(img, params[0], params[1], params[3], params[4])



# def augment(self, distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=0):

class Rotation(StochasticAugmenter):
    def __init__( self, degrees, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0, *args, **kwargs):
        super(Rotation, self).__init__(*args, **kwargs)
        if fill == -1:
            fill = list((255 * torch.rand(3)).numpy().astype(np.uint8))
        self.r = T.RandomRotation(degrees, interpolation=interpolation, expand=expand, center=center, fill=fill)

    def augment(self, image):
        image = torch.tensor(image).permute(2, 0, 1)
        params = (self.r.get_params(self.r.degrees),self.r.resample, self.r.expand, self.r.center, self.r.fill)
        return self.Rotation(image, params).permute(1,2,0).numpy(), params


    @staticmethod
    def aug(img,params):
        return F.rotate(img, params[0], params[1], params[2], params[3], params[4])

class Flip(Augmenter):
    def __init__( self, axis=(0,), *args, **kwargs):
        super(Flip, self).__init__(*args, **kwargs)
        assert len(axis) <= 3
        # Have to remap axises, to make it consistent with the otherones, since most of the pytorch ones require a channel, h, w format
        #  You can "Flip" the colors if you want too, why the hell not.
        temp = []
        for x in axis:
            if x == 0:
                temp.append(1)
            elif x == 1:
                temp.append(2)
            elif x == 3:
                temp.append(0)
        self.axis = tuple(axis)

    def augment(self, image):
        image = torch.tensor(image).permute(2, 0, 1)
        params = self.axis
        return self.flip(image,params).permute(1,2,0).numpy(), params


    @staticmethod
    def aug(img,params):
        return torch.flip(img, dims=params)


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
