from contextlib import nullcontext

import numpy as np
from loguru import logger

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import load_data_with_context, load_image
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter

torch = LazyLoader("torch")

OP_NAME = "image_document_domain_filter"

# ═══════════════════════════════════════════════════════════════════════════
# SigLIP 2 zero-shot anchor prompts
#
# Positive anchors mirror BizGenEval's document domains (slide / chart /
# webpage / poster / scientific figure / infographic); negatives are common
# web-crawl categories that are NOT document-style images.  Each class carries
# English AND Chinese phrase variants; per-language templates are applied and
# ALL resulting embeddings are averaged into ONE anchor vector per class, so
# scoring stays a simple (n_domains + n_negatives)-way softmax.
# ═══════════════════════════════════════════════════════════════════════════
TEMPLATES = {
    "en": ["This is a photo of {}.", "an image of {}", "{}"],
    "zh": ["{}", "这是{}。"],
}

DOMAIN_CLASSES = {
    "slide": {
        "en": ["a presentation slide with text", "a PowerPoint slide"],
        "zh": ["一页带文字的演示幻灯片", "一页PPT"],
    },
    "chart": {
        "en": [
            "a data chart with axis labels",
            "a bar chart, line graph, pie chart, scatter plot, or area chart",
            "a statistical graph or plot",
            "a data visualization diagram",
        ],
        "zh": ["一张带坐标轴标签的数据图表", "一张柱状图、折线图、饼图或散点图", "一张统计图表", "一张数据可视化图表"],
    },
    "webpage": {
        "en": ["a screenshot of a webpage", "a landing page design with text"],
        "zh": ["一张网页截图", "一个带文字的网页设计稿"],
    },
    "poster": {
        "en": ["a poster with text", "an advertising poster design"],
        "zh": ["一张带文字的海报", "一张广告海报设计"],
    },
    "sci_figure": {
        "en": [
            "a scientific figure from a research paper",
            "a labeled technical diagram",
            "a scientific schematic diagram",
            "a schematic illustration of a process or mechanism",
        ],
        "zh": ["一幅论文中的科研插图", "一幅带标注的技术示意图", "一幅科学示意图", "一幅原理示意图", "一张流程示意图", "一幅结构示意图"],
    },
    "infographic": {
        "en": ["an infographic"],
        "zh": ["一张信息图"],
    },
}

NEGATIVE_CLASSES = {
    "photo": {
        "en": ["a photograph", "a landscape photo", "a close-up photo", "a natural outdoor scene", "a photo of an object"],
        "zh": ["一幅照片", "一张风景照片", "一张物体的照片"],
    },
    "portrait": {
        "en": ["a portrait of a person"],
        "zh": ["一张人物肖像照片"],
    },
    "plain_text": {
        "en": ["a scanned text document", "a page of plain text"],
        "zh": ["一页扫描的文字文档", "一页纯文字页面"],
    },
    "anime_art": {
        "en": ["anime artwork", "a digital illustration of a character"],
        "zh": ["一幅动漫插画"],
    },
    "game_screenshot": {
        "en": ["a video game screenshot"],
        "zh": ["一张电子游戏画面截图"],
    },
    "meme_comic": {
        "en": ["a meme with overlaid text", "a comic page"],
        "zh": ["一张表情包", "一页漫画"],
    },
    "chat_screenshot": {
        "en": ["a screenshot of a chat conversation in a messaging app"],
        "zh": ["一张聊天软件对话截图"],
    },
}

DOMAIN_NAMES = list(DOMAIN_CLASSES.keys())
N_POS = len(DOMAIN_NAMES)

# domain name -> the StatsKeys field that holds its per-image softmax scores.
# Resolved eagerly so a missing/renamed constant fails loudly at import time.
DOMAIN_SCORE_KEYS = {d: getattr(StatsKeys, f"image_domain_score_{d}") for d in DOMAIN_NAMES}


def _class_prompts(cls):
    """Expand a {lang -> phrases} class into the full templated prompt list."""
    prompts = []
    for lang, phrases in cls.items():
        for tpl in TEMPLATES.get(lang, ["{}"]):
            for p in phrases:
                prompts.append(tpl.format(p))
    return prompts


def _is_naflex_ckpt(ckpt):
    """True for NaFlex checkpoints (aspect-ratio-preserving, variable
    resolution), e.g. ``google/siglip2-so400m-patch16-naflex``.  NaFlex accepts
    a ``max_num_patches`` kwarg on the image processor and emits
    ``pixel_attention_mask`` + ``spatial_shapes``; FixRes bakes a fixed square
    into the checkpoint and rejects ``max_num_patches``."""
    return "naflex" in ckpt.lower()


@OPERATORS.register_module(OP_NAME)
class ImageDocumentDomainFilter(Filter):
    """Filter to keep images that look like document-style graphics
    (slides / charts / webpages / posters / scientific figures / infographics)
    using SigLIP 2 zero-shot classification.

    For each image a ``document_domain_score`` is computed as the summed softmax
    mass over the positive document domains, against a closed set of positive
    domain anchors plus negative (photo / portrait / anime / game / meme / chat)
    anchors.  Samples are kept when the score falls within
    ``[min_score, max_score]``.  With multiple images, the 'any' strategy keeps
    the sample if any image qualifies (mirroring the standalone scorer's
    "best image wins"), while 'all' requires every image to qualify.

    Cached stats (all per image, one entry per image in the sample):
      * ``image_document_domain_score`` -- the biz score (summed positive mass),
        used for keep/drop;
      * ``image_document_domain`` -- the argmax positive domain label;
      * ``image_domain_score_<domain>`` -- the individual softmax probability of
        each of the 6 positive domains (slide / chart / webpage / poster /
        sci_figure / infographic)."""

    _accelerator = "cuda"
    _batched_op = True

    def __init__(
        self,
        hf_siglip: str = "google/siglip2-so400m-patch16-naflex",
        trust_remote_code: bool = False,
        max_num_patches: int = 1024,
        min_score: float = 0.35,
        max_score: float = 1.0,
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_siglip: SigLIP 2 model name on huggingface used for zero-shot
            document-domain scoring. Both NaFlex (aspect-ratio-preserving) and
            FixRes checkpoints are supported; the variant is detected by name.
        :param trust_remote_code: whether to trust the remote code of HF models.
        :param max_num_patches: only used for NaFlex checkpoints — controls
            resolution (higher = sharper, more compute). Ignored for FixRes.
        :param min_score: min document-domain score to keep a sample.
        :param max_score: max document-domain score to keep a sample.
        :param any_or_all: keep this sample with 'any' or 'all' strategy over
            all images. 'any': keep if any image meets the condition. 'all':
            keep only if all images meet the condition.
        :param args: extra args
        :param kwargs: extra args
        """
        kwargs["memory"] = "4000MB" if kwargs.get("memory", 0) == 0 else kwargs["memory"]
        super().__init__(*args, **kwargs)
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"
        self.min_score = min_score
        self.max_score = max_score

        self.model_key = prepare_model(
            model_type="huggingface",
            pretrained_model_name_or_path=hf_siglip,
            trust_remote_code=trust_remote_code,
        )
        # NaFlex needs max_num_patches on the image processor call; FixRes {}.
        self.proc_kwargs = {"max_num_patches": max_num_patches} if _is_naflex_ckpt(hf_siglip) else {}
        # Text anchors are expensive (a text-tower forward over ~60 prompts) and
        # device-bound, so build them once per rank and cache. Keyed by rank
        # because in multi-GPU runs each rank holds the model on its own device.
        self._anchor_cache = {}

    @torch.no_grad()
    def _get_anchors(self, model, processor):
        """Return ``(text_feats, logit_scale)``: one L2-normalised averaged
        anchor per class, positive domains first (order = ``DOMAIN_NAMES``),
        then negatives. Cached per device."""
        device = model.device
        key = str(device)
        if key in self._anchor_cache:
            return self._anchor_cache[key]

        def _encode_class(cls):
            # .lower() is a no-op for Chinese and matches SigLIP's lowercased
            # training text.
            prompts = [p.lower() for p in _class_prompts(cls)]
            inputs = processor(
                text=prompts,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            f = model.get_text_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.mean(dim=0)
            return f / f.norm()

        feats = [_encode_class(c) for c in DOMAIN_CLASSES.values()]
        feats += [_encode_class(c) for c in NEGATIVE_CLASSES.values()]
        text_feats = torch.stack(feats)
        logit_scale = model.logit_scale.exp().item()
        self._anchor_cache[key] = (text_feats, logit_scale)
        return text_feats, logit_scale

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.image_document_domain_score in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_document_domain_score] = np.array([], dtype=np.float64)
            sample[Fields.stats][StatsKeys.image_document_domain] = []
            for d in DOMAIN_NAMES:
                sample[Fields.stats][DOMAIN_SCORE_KEYS[d]] = np.array([], dtype=np.float64)
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        model, processor = get_model(self.model_key, rank, self.use_cuda())
        text_feats, logit_scale = self._get_anchors(model, processor)

        inputs = processor(images=list(images.values()), return_tensors="pt", **self.proc_kwargs).to(model.device)
        # bf16 autocast on GPU for the image tower (much faster on Ampere+),
        # cast similarities back to fp32 for a stable softmax.
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if self.use_cuda() else nullcontext()
        with torch.no_grad(), autocast_ctx:
            feats = model.get_image_features(**inputs)
            # eps-guarded normalize: a degenerate all-zero feature would
            # otherwise produce NaNs that silently poison the score.
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            sims = feats.float() @ text_feats.float().T  # unscaled cosine
            probs = (logit_scale * sims).softmax(dim=-1)
            dom_probs = probs[:, :N_POS].cpu().numpy()  # (n_images, N_POS)

        scores = dom_probs.sum(axis=-1)  # summed positive-domain mass (biz score)
        domains = [DOMAIN_NAMES[int(i)] for i in dom_probs.argmax(axis=-1)]

        sample[Fields.stats][StatsKeys.image_document_domain_score] = [float(s) for s in scores]
        sample[Fields.stats][StatsKeys.image_document_domain] = domains
        # per-domain softmax probabilities, one list (per image) per domain
        for j, d in enumerate(DOMAIN_NAMES):
            sample[Fields.stats][DOMAIN_SCORE_KEYS[d]] = [float(p) for p in dom_probs[:, j]]
        logger.debug(f"document_domain_scores: {scores}, domains: {domains}")
        return sample

    def process_single(self, sample):
        scores = sample[Fields.stats][StatsKeys.image_document_domain_score]
        if len(scores) <= 0:
            return True

        keep_bools = np.array([self.get_keep_boolean(s, self.min_score, self.max_score) for s in scores])

        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
