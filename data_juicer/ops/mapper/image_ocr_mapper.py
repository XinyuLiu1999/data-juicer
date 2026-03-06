import numpy as np

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ..base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_IMAGES

OP_NAME = "image_ocr_mapper"

paddleocr = LazyLoader("paddleocr")


def _to_serializable(obj):
    """Convert numpy arrays and other non-serializable types to plain Python
    types for JSON-compatible output."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]
    return obj


@UNFORKABLE.register_module(OP_NAME)
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageOCRMapper(Mapper):
    """Perform OCR on images using PP-OCRv5 and store the results in metadata.

    This operator uses PaddleOCR (PP-OCRv5) to detect and recognize text in
    images. It processes each image in the sample and stores the full OCR
    results in the sample's metadata under the `image_ocr_tag` key, along
    with an OCR difficulty score under the `image_ocr_difficulty` key.

    Each OCR result is a dict containing keys such as: `dt_polys`,
    `rec_texts`, `rec_scores`, `rec_polys`, `rec_boxes`,
    `text_det_params`, `text_type`, `textline_orientation_angles`, etc.

    The difficulty score (0.0 to 1.0) is computed per image based on:
    - Text density: fraction of image area covered by text bounding boxes.
    - Number of boxes: more boxes indicate more complex layout.
    - Low-confidence ratio: fraction of boxes with recognition confidence
      below `low_conf_thresh`.
    """

    _accelerator = "cpu"

    def __init__(
        self,
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        low_conf_thresh=0.5,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param lang: language for OCR recognition (e.g. 'en', 'ch').
        :param use_doc_orientation_classify: whether to use document
            orientation classification.
        :param use_doc_unwarping: whether to use document unwarping.
        :param use_textline_orientation: whether to use textline orientation.
        :param text_detection_model_name: name of the text detection model.
        :param text_recognition_model_name: name of the text recognition model.
        :param low_conf_thresh: confidence threshold below which a text box
            is considered low-confidence for difficulty calculation.
        """
        kwargs.setdefault("memory", "2GB")
        super().__init__(*args, **kwargs)

        self.lang = lang
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_textline_orientation = use_textline_orientation
        self.text_detection_model_name = text_detection_model_name
        self.text_recognition_model_name = text_recognition_model_name
        self.low_conf_thresh = low_conf_thresh
        self._model = None

    def _get_model(self, rank=None):
        if self._model is None:
            device = None
            if self.use_cuda():
                device = f"gpu:{rank}" if rank is not None else "gpu:0"
            self._model = paddleocr.PaddleOCR(
                lang=self.lang,
                use_doc_orientation_classify=self.use_doc_orientation_classify,
                use_doc_unwarping=self.use_doc_unwarping,
                use_textline_orientation=self.use_textline_orientation,
                text_detection_model_name=self.text_detection_model_name,
                text_recognition_model_name=self.text_recognition_model_name,
                device=device,
            )
        return self._model

    def _compute_difficulty(self, ocr_page_results, image_width, image_height):
        """Compute an OCR difficulty score (0.0-1.0) for a single image.

        The score is the average of three components:
        - text_density: fraction of image area covered by text bounding boxes,
          clamped to [0, 1].
        - box_count_score: normalized number of text boxes
          (n_boxes / (n_boxes + 10)), saturates toward 1 for many boxes.
        - low_conf_ratio: fraction of boxes with confidence < low_conf_thresh.
        """
        image_area = image_width * image_height
        if image_area == 0:
            return 0.0

        all_boxes = []
        all_scores = []
        for res in ocr_page_results:
            all_boxes.extend(res.get("rec_boxes", []))
            all_scores.extend(res.get("rec_scores", []))

        n_boxes = len(all_boxes)
        if n_boxes == 0:
            return 0.0

        # text density: total box area / image area
        total_box_area = 0.0
        for box in all_boxes:
            x1, y1, x2, y2 = box
            total_box_area += abs(x2 - x1) * abs(y2 - y1)
        text_density = min(total_box_area / image_area, 1.0)

        # box count score: saturating function
        box_count_score = n_boxes / (n_boxes + 10.0)

        # low confidence ratio
        n_low_conf = sum(1 for s in all_scores if s < self.low_conf_thresh)
        low_conf_ratio = n_low_conf / n_boxes

        return (text_density + box_count_score + low_conf_ratio) / 3.0

    def process_single(self, sample, rank=None, context=False):
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.meta][MetaKeys.image_ocr_tag] = []
            sample[Fields.meta][MetaKeys.image_ocr_difficulty] = []
            return sample

        if MetaKeys.image_ocr_tag in sample[Fields.meta]:
            return sample

        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample,
            context,
            loaded_image_keys,
            load_image,
            mm_bytes_key=self.image_bytes_key,
        )

        model = self._get_model(rank=rank)
        ocr_results = []
        difficulty_scores = []

        for image_key in loaded_image_keys:
            image = images[image_key]
            image_array = np.array(image)
            result = model.predict(image_array)

            image_ocr = []
            for res in result:
                image_ocr.append(_to_serializable(res))
            ocr_results.append(image_ocr)

            w, h = image.size
            difficulty = self._compute_difficulty(image_ocr, w, h)
            difficulty_scores.append(difficulty)

        sample[Fields.meta][MetaKeys.image_ocr_tag] = ocr_results
        sample[Fields.meta][MetaKeys.image_ocr_difficulty] = difficulty_scores
        return sample
