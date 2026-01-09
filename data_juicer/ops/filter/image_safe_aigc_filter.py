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

    The model outputs a probability score where higher values indicate
    a higher likelihood that the image is AI-generated (fake).
    """

    _accelerator = "cuda"

    def __init__(
        self,
        checkpoint_path: str = "",
        min_score: float = 0.0,
        max_score: float = 0.5,
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
        :param min_score: Minimum AIGC score threshold. Images with scores
            below this will be filtered out. Range from 0 to 1. Default is 0.
        :param max_score: Maximum AIGC score threshold. Images with scores
            above this will be filtered out. Range from 0 to 1. Default is 0.5
            (keeping images that are more likely to be real).
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
        self.min_score = min_score
        self.max_score = max_score

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
                [], dtype=np.float64
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

        # process images and compute AIGC scores
        aigc_scores = []
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
                # SAFE outputs logits, apply softmax to get probability
                # Class 1 is typically the "fake/AIGC" class
                if output.dim() == 1:
                    # Single output (probability of being fake)
                    prob = torch.sigmoid(output).item()
                else:
                    # Two-class output [real, fake]
                    prob = torch.softmax(output, dim=-1)[0, 1].item()
                aigc_scores.append(float(prob))

        sample[Fields.stats][StatsKeys.image_aigc_score] = aigc_scores

        return sample

    def process_single(self, sample, rank=None):
        aigc_scores = sample[Fields.stats][StatsKeys.image_aigc_score]
        if len(aigc_scores) <= 0:
            return True

        keep_bools = np.array(
            [
                self.get_keep_boolean(score, self.min_score, self.max_score)
                for score in aigc_scores
            ]
        )

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
