# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from PIL import Image
from rembg import remove, new_session


class BackgroundRemover():
    def __init__(self, model_name='u2net'):
        self.session = new_session(model_name=model_name)

    def __call__(self, image: Image.Image, am=False, amft=240, ambt=10, ames=10):
        """
        Remove background from image.
        
        Parameters
        ----------
        image : Image
            Input image.
        am : bool, optional, default=False
            Alpha matting.
        amft : int, optional, default=240
            Alpha matting foreground threshold.
        ambt : int, optional, default=10
            Alpha matting background threshold.
        ames : int, optional, default=10
            Alpha matting erode size.
        
        Returns
        -------
        output : btyes | Image | ndarray
            Cutout image with background removed.
        """

        output = remove(
            image, alpha_matting=am, alpha_matting_foreground_threshold=amft, alpha_matting_background_threshold=ambt, alpha_matting_erode_size=ames,
            session=self.session, bgcolor=(255, 255, 255, 0),
        )
        return output
