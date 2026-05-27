import numpy as np

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ..base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_IMAGES

OP_NAME = "image_ocr_mapper"

paddleocr = LazyLoader("paddleocr")


@UNFORKABLE.register_module(OP_NAME)
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageOCRMapper(Mapper):
    """Extract text from images with PP-OCRv5 and store a region-grouped
    string plus a difficulty score in metadata.

    For each image the mapper writes two entries under ``sample[Fields.meta]``:

    - ``image_ocr_text``: a plain string with high-confidence text grouped
      into ``top``/``middle``/``bottom`` thirds of the image (by box
      y-center) and joined with " | "; regions are separated by newlines.
      Boxes below ``conf_thresh`` are dropped. Empty regions are omitted;
      an image with no high-confidence text becomes ``""``.
    - ``image_ocr_difficulty``: a float in ``[0.0, 1.0]`` averaging text
      density, box-count saturation, and low-confidence ratio (boxes below
      ``low_conf_thresh``).

    Example ``image_ocr_text`` output::

        top: HEADLINE | SUBHEAD
        middle: body text snippet
        bottom: footer line

    Both outputs are flat scalars / strings, sidestepping the PyArrow
    schema-unification failures that the previous deeply-nested struct
    output triggered when Ray Data concatenated heterogeneous batches.
    """

    _accelerator = "cpu"

    def __init__(
        self,
        lang=None,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        conf_thresh=0.8,
        low_conf_thresh=0.5,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param lang: language for OCR recognition. ``None`` (default) uses
            PP-OCRv5's built-in multilingual recognizer (CN/EN/JP/pinyin).
            Set to e.g. ``'en'`` or ``'ch'`` to force a language-specific
            recognizer.
        :param use_doc_orientation_classify: whether to use document
            orientation classification.
        :param use_doc_unwarping: whether to use document unwarping.
        :param use_textline_orientation: whether to use textline orientation.
        :param text_detection_model_name: name of the text detection model.
        :param text_recognition_model_name: name of the text recognition
            model.
        :param conf_thresh: drop recognized boxes with confidence below
            this value before grouping into top/middle/bottom regions.
        :param low_conf_thresh: confidence threshold below which a text box
            counts as "low-confidence" in the difficulty score (separate
            from ``conf_thresh`` which gates the text output).
        """
        kwargs.setdefault("memory", "2GB")
        super().__init__(*args, **kwargs)

        self.lang = lang
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_textline_orientation = use_textline_orientation
        self.text_detection_model_name = text_detection_model_name
        self.text_recognition_model_name = text_recognition_model_name
        self.conf_thresh = conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self._model = None

    def _get_model(self, rank=None):
        if self._model is None:
            device = None
            if self.use_cuda():
                device = f"gpu:{rank}" if rank is not None else "gpu:0"
            kwargs = dict(
                use_doc_orientation_classify=self.use_doc_orientation_classify,
                use_doc_unwarping=self.use_doc_unwarping,
                use_textline_orientation=self.use_textline_orientation,
                text_detection_model_name=self.text_detection_model_name,
                text_recognition_model_name=self.text_recognition_model_name,
                device=device,
            )
            if self.lang is not None:
                kwargs["lang"] = self.lang
            self._model = paddleocr.PaddleOCR(**kwargs)
        return self._model

    def _format_regions(self, res, image_height: int) -> str:
        """Group recognized text by vertical thirds of the image."""
        if image_height <= 0:
            return ""

        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []
        polys = res.get("rec_polys") or []

        regions = {"top": [], "middle": [], "bottom": []}
        for text, conf, poly in zip(texts, scores, polys):
            if conf is None or conf < self.conf_thresh:
                continue
            if not poly:
                continue
            y_center = sum(p[1] for p in poly) / len(poly)
            ratio = y_center / image_height
            if ratio < 1 / 3:
                region = "top"
            elif ratio < 2 / 3:
                region = "middle"
            else:
                region = "bottom"
            regions[region].append(text)

        parts = []
        for region in ("top", "middle", "bottom"):
            if regions[region]:
                parts.append(f"{region}: {' | '.join(regions[region])}")
        return "\n".join(parts)

    def _compute_difficulty(self, res, image_width: int, image_height: int) -> float:
        """Compute an OCR difficulty score (0.0-1.0) for a single image.

        Average of three components:
        - text_density: fraction of image area covered by text boxes,
          clamped to [0, 1].
        - box_count_score: n_boxes / (n_boxes + 10), saturating toward 1.
        - low_conf_ratio: fraction of boxes with confidence < low_conf_thresh.
        """
        image_area = image_width * image_height
        if image_area == 0:
            return 0.0

        boxes = res.get("rec_boxes") or []
        scores = res.get("rec_scores") or []
        n_boxes = len(boxes)
        if n_boxes == 0:
            return 0.0

        total_box_area = 0.0
        for box in boxes:
            if box is None or len(box) < 4:
                continue
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            total_box_area += abs(x2 - x1) * abs(y2 - y1)
        text_density = min(total_box_area / image_area, 1.0)

        box_count_score = n_boxes / (n_boxes + 10.0)

        n_low_conf = sum(
            1 for s in scores if s is not None and s < self.low_conf_thresh
        )
        low_conf_ratio = n_low_conf / n_boxes

        return (text_density + box_count_score + low_conf_ratio) / 3.0

    def process_single(self, sample, rank=None, context=False):
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.meta][MetaKeys.image_ocr_text] = []
            sample[Fields.meta][MetaKeys.image_ocr_difficulty] = []
            return sample

        if MetaKeys.image_ocr_text in sample[Fields.meta]:
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
        ocr_texts = []
        difficulty_scores = []

        for image_key in loaded_image_keys:
            image = images[image_key]
            try:
                results = model.predict(np.array(image))
            except Exception:
                ocr_texts.append("")
                difficulty_scores.append(0.0)
                continue

            if not results:
                ocr_texts.append("")
                difficulty_scores.append(0.0)
                continue

            res = results[0]
            w, h = image.size
            ocr_texts.append(self._format_regions(res, h))
            difficulty_scores.append(self._compute_difficulty(res, w, h))

        sample[Fields.meta][MetaKeys.image_ocr_text] = ocr_texts
        sample[Fields.meta][MetaKeys.image_ocr_difficulty] = difficulty_scores
        return sample
