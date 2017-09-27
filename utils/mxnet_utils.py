import PIL
import numpy as np
import mxnet as mx
from mxnet import rnn
from mxnet import image as img

class NLP(object):

    @staticmethod
    def encode_sentences(sentences, vocab=None, invalid_label=-1, invalid_key='\n', start_label=0):
        """Encode sentences and (optionally) build a mapping
        from string tokens to integer indices. Unknown keys
        will be added to vocabulary.

        Parameters
        ----------
        sentences : list of list of str
            A list of sentences to encode. Each sentence
            should be a list of string tokens.
        vocab : None or dict of str -> int
            Optional input Vocabulary
        invalid_label : int, default -1
            Index for invalid token, like <end-of-sentence>
        invalid_key : str, default '\\n'
            Key for invalid token. Use '\\n' for end
            of sentence by default.
        start_label : int
            lowest index.

        Returns
        -------
        result : list of list of int
            encoded sentences
        vocab : dict of str -> int
            result vocabulary
        """
        return rnn.io.encode_sentences(sentences, vocab, invalid_label, invalid_key, start_label)


class Image(object):

    @staticmethod
    def read(filename, flag=1, to_rgb=True, out=None):
        """Read and decode an image to an NDArray.

        Note: `imread` uses OpenCV (not the CV2 Python library).
        MXNet must have been built with USE_OPENCV=1 for `imdecode` to work.

        Parameters
        ----------
        filename : str
            Name of the image file to be loaded.
        flag : {0, 1}, default 1
            1 for three channel color output. 0 for grayscale output.
        to_rgb : bool, default True
            True for RGB formatted output (MXNet default).
            False for BGR formatted output (OpenCV default).
        out : NDArray, optional
            Output buffer. Use `None` for automatic allocation.

        Returns
        -------
        NDArray
            An `NDArray` containing the image.

        Example
        -------
        >>> image.read("flower.jpg", flag=0, to_rgb=0)
        <NDArray 224x224x3 @cpu(0)>
        """
        img_arr = img.imread(filename, flag, to_rgb, out)
        img_arr = mx.nd.transpose(img_arr, (2,0,1))
        img_arr = mx.nd.expand_dims(img_arr, axis=0)
        return img_arr

    @staticmethod
    def write(filename, img_arr, flag=1):
        """Write an NDArray

        Parameters
        ----------
        filename : str
            Name of the image file to be written to.
        img_arr : NDArray
            Image in NDArray format with shape (channel, width, height)
        flag : {0, 1}, default 1
            1 for three channel color output. 0 for grayscale output.
        """
        img_arr = mx.nd.transpose(img_arr, (1,2,0)).astype(np.uint8).asnumpy()
        mode = 'RGB' if flag == 1 else 'L'
        output = PIL.Image.fromarray(img_arr, mode)
        output.save(filename)

    @staticmethod
    def resize(src, new_width, new_height, interp=2):
        """Read and decode an image to an NDArray.

        Note: `imread` uses OpenCV (not the CV2 Python library).
        MXNet must have been built with USE_OPENCV=1 for `imdecode` to work.

        Parameters
        ----------
        src : NDArray
            Source image in NDArray format
        new_width : int
            Width in pixel for resized image
        new_height : int
            Height in pixel for resized image
        interp : int
            interpolation method for all resizing operations

            Possible values:
            0: Nearest Neighbors Interpolation.
            1: Bilinear interpolation.
            2: Area-based (resampling using pixel area relation). It may be a
            preferred method for image decimation, as it gives moire-free
            results. But when the image is zoomed, it is similar to the Nearest
            Neighbors method. (used by default).
            3: Bicubic interpolation over 4x4 pixel neighborhood.
            4: Lanczos interpolation over 8x8 pixel neighborhood.
            9: Cubic for enlarge, area for shrink, bilinear for others
            10: Random select from interpolation method metioned above.
            Note:
            When shrinking an image, it will generally look best with AREA-based
            interpolation, whereas, when enlarging an image, it will generally look best
            with Bicubic (slow) or Bilinear (faster but still looks OK).
            More details can be found in the documentation of OpenCV, please refer to
            http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.

        Returns
        -------
        NDArray
            An `NDArray` containing the resized image.
        """
        return img.imresize(src, new_width, new_height, interp)

    @staticmethod
    def fixed_crop(src, x0, y0, w, h, size=None, interp=2):
        """Crop src at fixed location, and (optionally) resize it to size.

        Parameters
        ----------
        src : NDArray
            Input image
        x0 : int
            Left boundary of the cropping area
        y0 : int
            Top boundary of the cropping area
        w : int
            Width of the cropping area
        h : int
            Height of the cropping area
        size : tuple of (w, h)
            Optional, resize to new size after cropping
        interp : int, optional, default=2
            Interpolation method. See resize for details.

        Returns
        -------
        NDArray
            An `NDArray` containing the cropped image.
        """
        return img.fixed_crop(src, x0, y0, w, h, size, interp)

    @staticmethod
    def color_normalize(src, mean, std=None):
        """Normalize src with mean and std.

        Parameters
        ----------
        src : NDArray
            Input image
        mean : NDArray
            RGB mean to be subtracted
        std : NDArray
            RGB standard deviation to be divided

        Returns
        -------
        NDArray
            An `NDArray` containing the normalized image.
        """
        return img.color_normalize(src, mean, std)

