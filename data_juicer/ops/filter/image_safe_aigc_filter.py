import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import load_data_with_context, load_image
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

torch = LazyLoader("torch")

OP_NAME = "image_safe_aigc_filter"


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageSafeAigcFilter(Filter):
    """Filter to keep samples whose images are detected as real (non-AIGC)
    images using the SAFE model.

    SAFE (Synthetic AI-generated image Filter with Enhanced generalization)
    is a model designed to distinguish between real and AI-generated images
    with high generalization across different generation methods including
    diffusion models, autoregressive models, and closed-source generators.

    Reference: https://github.com/Ouxiang-Li/SAFE
    Paper: "Improving Synthetic Image Detection Towards Generalization:
           An Image Transformation Perspective" (KDD2025)

    The model outputs a binary prediction:
    - 0: Real image
    - 1: AI-generated (fake) image

    By default, this filter keeps real images (prediction=0) and removes
    AI-generated images (prediction=1).
    """

    _accelerator = "cuda"

    def __init__(
        self,
        checkpoint_path: str = "",
        keep_real: bool = True,
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param checkpoint_path: Path to the SAFE model checkpoint file.
            If empty, will attempt to use the default checkpoint location
            in the cloned SAFE repository. Users should download the
            checkpoint from https://github.com/Ouxiang-Li/SAFE.
        :param keep_real: If True (default), keep real images and filter out
            AI-generated images. If False, keep AI-generated images and
            filter out real images.
        :param any_or_all: Strategy for handling multiple images in a sample.
            'any': keep the sample if any image meets the condition.
            'all': keep the sample only if all images meet the condition.
        :param args: Extra positional arguments.
        :param kwargs: Extra keyword arguments.
        """
        kwargs["memory"] = (
            "2GB" if kwargs.get("memory", 0) == 0 else kwargs["memory"]
        )
        super().__init__(*args, **kwargs)
        self.keep_real = keep_real

        if any_or_all not in ["any", "all"]:
            raise ValueError(
                f"Keep strategy [{any_or_all}] is not supported. "
                f'Can only be one of ["any", "all"].'
            )
        self.any = any_or_all == "any"

        self.model_key = prepare_model(
            model_type="safe",
            checkpoint_path=checkpoint_path,
        )

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.image_aigc_score in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_aigc_score] = np.array(
                [], dtype=np.int64
            )
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample,
            context,
            loaded_image_keys,
            load_image,
            mm_bytes_key=self.image_bytes_key,
        )

        # get model and transform
        model, transform = get_model(self.model_key, rank, self.use_cuda())

        # determine device
        device = next(model.parameters()).device

        # process images and compute binary predictions
        predictions = []
        for key in loaded_image_keys:
            image = images[key]
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Apply transform and add batch dimension
            input_tensor = transform(image).unsqueeze(0).to(device)

            # Run inference
            with torch.no_grad():
                output = model(input_tensor)
                # Binary prediction: 0=real, 1=fake
                if output.dim() == 1 or output.shape[-1] == 1:
                    # Single logit output
                    pred = int(output.squeeze() > 0)
                else:
                    # Two-class output [real, fake]
                    pred = int(output.argmax(dim=-1).item())
                predictions.append(pred)

        sample[Fields.stats][StatsKeys.image_aigc_score] = predictions

        return sample

    def process_single(self, sample, rank=None):
        predictions = sample[Fields.stats][StatsKeys.image_aigc_score]
        if len(predictions) <= 0:
            return True

        # Determine which prediction value to keep
        # keep_real=True: keep prediction=0 (real)
        # keep_real=False: keep prediction=1 (fake)
        target_pred = 0 if self.keep_real else 1

        keep_bools = np.array(
            [pred == target_pred for pred in predictions]
        )

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
