import os
from typing import List, Optional, Union

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from jsonargparse import Namespace, dict_to_namespace
from loguru import logger

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import find_files_with_suffix, is_absolute_path
from data_juicer.utils.registry import Registry

FORMATTERS = Registry("Formatters")


class BaseFormatter:
    """Base class to load dataset."""

    def load_dataset(self, *args) -> Dataset:
        raise NotImplementedError


class LocalFormatter(BaseFormatter):
    """The class is used to load a dataset from local files or local
    directory."""

    def __init__(
        self,
        dataset_path: str,
        type: str,
        suffixes: Union[str, List[str], None] = None,
        text_keys: List[str] = None,
        add_suffix=False,
        laioncoco_preprocessing=False,
        blip3o_preprocessing=False,
        taisu_preprocessing=False,
        danqing_preprocessing=False,
        **kwargs,
    ):
        """
        Initialization method.

        :param dataset_path: path to a dataset file or a dataset
            directory
        :param type: a packaged dataset module type (json, csv, etc.)
        :param suffixes: files with specified suffixes to be processed
        :param text_keys: key names of field that stores sample
            text.
        :param add_suffix: whether to add the file suffix to dataset
            meta info
        :param laioncoco_preprocessing: whether to apply LAION-COCO
            format preprocessing
        :param blip3o_preprocessing: whether to apply BLIP3o WebDataset
            format preprocessing
        :param taisu_preprocessing: whether to apply TaiSu (image-only
            WebDataset) format preprocessing
        :param danqing_preprocessing: whether to apply DanQing (image+caption
            parquet) format preprocessing
        :param kwargs: extra args
        """
        self.type = type
        self.kwargs = kwargs
        self.text_keys = text_keys
        self.data_files = find_files_with_suffix(dataset_path, suffixes)
        self.add_suffix = add_suffix
        self.laioncoco_preprocessing = laioncoco_preprocessing
        self.blip3o_preprocessing = blip3o_preprocessing
        self.taisu_preprocessing = taisu_preprocessing
        self.danqing_preprocessing = danqing_preprocessing

    def load_dataset(self, num_proc: Optional[int] = None, global_cfg=None) -> Dataset:
        """
        Load a dataset from dataset file or dataset directory, and unify its
        format.

        :param num_proc: number of processes when loading the dataset
        :param global_cfg: global cfg used in consequent processes,
        :return: formatted dataset
        """
        _num_proc = self.kwargs.pop("num_proc", 1)
        num_proc = num_proc or _num_proc
        logger.info(f"Loading dataset with num_proc: {num_proc}")
        datasets = load_dataset(
            self.type,
            data_files={key.strip("."): self.data_files[key] for key in self.data_files},
            num_proc=num_proc,
            **self.kwargs,
        )
        if self.add_suffix:
            logger.info("Add suffix info into dataset...")
            datasets = add_suffixes(datasets, num_proc)
        else:
            from data_juicer.core.data import NestedDataset

            datasets = NestedDataset(concatenate_datasets([ds for _, ds in datasets.items()]))
        if self.laioncoco_preprocessing:
            datasets = preprocess_laioncoco(datasets, num_proc)
        if self.blip3o_preprocessing:
            datasets = preprocess_blip3o(datasets, num_proc)
        if self.taisu_preprocessing:
            datasets = preprocess_taisu(datasets, num_proc)
        if self.danqing_preprocessing:
            datasets = preprocess_danqing(datasets, num_proc)
        ds = unify_format(datasets, text_keys=self.text_keys, num_proc=num_proc, global_cfg=global_cfg)
        return ds


class RemoteFormatter(BaseFormatter):
    """The class is used to load a dataset from repository of huggingface
    hub."""

    def __init__(self, dataset_path: str, text_keys: List[str] = None, **kwargs):
        """
        Initialization method.

        :param dataset_path: a dataset file or a dataset directory
        :param text_keys: key names of field that stores sample
            text.
        :param kwargs: extra args
        """
        self.path = dataset_path
        self.text_keys = text_keys
        self.kwargs = kwargs

    def load_dataset(self, num_proc: int = 1, global_cfg=None) -> Dataset:
        """
        Load a dataset from HuggingFace, and unify its format.

        :param num_proc: number of processes when loading the dataset
        :param global_cfg: the global cfg used in consequent processes,
        :return: formatted dataset
        """
        ds = load_dataset(self.path, split="train", num_proc=num_proc, **self.kwargs)
        ds = unify_format(ds, text_keys=self.text_keys, num_proc=num_proc, global_cfg=global_cfg)
        return ds


def add_suffixes(datasets: DatasetDict, num_proc: int = 1) -> Dataset:
    """
    Add suffix filed to datasets.

    :param datasets: a DatasetDict object
    :param num_proc: number of processes to add suffixes
    :return: datasets with suffix features.
    """
    logger.info("Add suffix column for dataset")
    from data_juicer.core.data import add_same_content_to_new_column

    for key, ds in datasets.items():
        if Fields.suffix not in ds.features:
            datasets[key] = ds.map(
                add_same_content_to_new_column,
                fn_kwargs={"new_column_name": Fields.suffix, "initial_value": "." + key},
                num_proc=num_proc,
                desc="Adding new column for suffix",
            )
    datasets = concatenate_datasets([ds for _, ds in datasets.items()])
    from data_juicer.core.data import NestedDataset

    return NestedDataset(datasets)


def preprocess_laioncoco(dataset, num_proc=1):
    """
    Preprocess LAION-COCO format dataset.

    Transforms the non-standard LAION-COCO schema into the format
    expected by Data-Juicer:
    - Extracts 'text' from the 'clean_content' JSON string, prepended
      with image special tokens so multimodal ops can correlate images
      with text
    - Flattens 'image_buffer_list' structs into 'images' (IDs) and
      'image_bytes' (raw bytes) columns

    :param dataset: a NestedDataset with LAION-COCO schema
    :param num_proc: number of processes for mapping
    :return: transformed NestedDataset
    """
    import json

    from data_juicer.core.data import NestedDataset
    from data_juicer.utils.mm_utils import SpecialTokens

    logger.info("Applying LAION-COCO preprocessing...")

    def transform(sample):
        # 1. Extract 'text' from clean_content JSON
        clean = json.loads(sample["clean_content"])
        text = clean.get("text", "")

        # 2. Flatten image_buffer_list into images (IDs) and
        #    image_bytes (raw bytes)
        buf_list = sample.get("image_buffer_list", []) or []
        sample["images"] = [item["image_id"] for item in buf_list]
        sample["image_bytes"] = [item["buffer"] for item in buf_list]

        # 3. Prepend image special tokens so multimodal ops can
        #    associate images with the text
        img_tokens = SpecialTokens.image * len(buf_list)
        sample["text"] = img_tokens + text

        return sample

    dataset = dataset.map(
        transform, num_proc=num_proc, desc="LAION-COCO preprocessing"
    )

    # Remove original complex columns that are no longer needed
    cols_to_remove = [
        col
        for col in ["clean_content", "image_buffer_list"]
        if col in dataset.column_names
    ]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)

    if not isinstance(dataset, NestedDataset):
        dataset = NestedDataset(dataset)
    return dataset


def preprocess_blip3o(dataset, num_proc=1):
    """
    Preprocess BLIP3o WebDataset format dataset.

    Transforms the BLIP3o WebDataset schema into the format
    expected by Data-Juicer:
    - Converts 'jpg' (PIL Image) to 'image_bytes' (raw JPEG bytes) and
      creates 'images' (synthetic IDs derived from __key__)
    - Renames 'txt' to 'text', prepended with image special token

    :param dataset: a NestedDataset with BLIP3o WebDataset schema
    :param num_proc: number of processes for mapping
    :return: transformed NestedDataset
    """
    import io

    from data_juicer.core.data import NestedDataset
    from data_juicer.utils.mm_utils import SpecialTokens

    logger.info("Applying BLIP3o WebDataset preprocessing...")

    def transform(sample):
        # Convert PIL image to bytes
        img = sample.get("jpg")
        key = sample.get("__key__", "unknown")
        if img is not None:
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            sample["image_bytes"] = [buf.getvalue()]
            sample["images"] = [f"{key}.jpg"]
        else:
            sample["image_bytes"] = []
            sample["images"] = []

        # Build text with image special token
        txt = sample.get("txt", "") or ""
        sample["text"] = SpecialTokens.image + txt

        return sample

    dataset = dataset.map(
        transform, num_proc=num_proc, desc="BLIP3o preprocessing"
    )

    # Remove original webdataset columns
    cols_to_remove = [
        col
        for col in ["jpg", "txt"]
        if col in dataset.column_names
    ]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)

    if not isinstance(dataset, NestedDataset):
        dataset = NestedDataset(dataset)
    return dataset


def preprocess_taisu(dataset, num_proc=1):
    """
    Preprocess TaiSu (image-only WebDataset) format dataset.

    Like preprocess_blip3o but for tars with NO caption ('jpg' + '__key__',
    no 'txt'):
    - Converts 'jpg' (PIL Image) to 'image_bytes' (raw JPEG bytes) and
      creates 'images' (synthetic IDs derived from __key__)
    - Sets 'text' to just the image special token (empty caption), so
      multimodal ops can still correlate the image with text. If a 'txt'
      field happens to be present its value is appended after the token.

    :param dataset: a NestedDataset with image-only WebDataset schema
    :param num_proc: number of processes for mapping
    :return: transformed NestedDataset
    """
    import io

    from data_juicer.core.data import NestedDataset
    from data_juicer.utils.mm_utils import SpecialTokens

    logger.info("Applying TaiSu (image-only WebDataset) preprocessing...")

    def transform(sample):
        img = sample.get("jpg")
        key = sample.get("__key__", "unknown")
        if img is not None:
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            sample["image_bytes"] = [buf.getvalue()]
            sample["images"] = [f"{key}.jpg"]
        else:
            sample["image_bytes"] = []
            sample["images"] = []

        # No caption in TaiSu: text is just the image special token.
        txt = sample.get("txt", "") or ""
        sample["text"] = SpecialTokens.image + txt

        return sample

    dataset = dataset.map(
        transform, num_proc=num_proc, desc="TaiSu preprocessing"
    )

    # Remove original webdataset columns
    cols_to_remove = [
        col
        for col in ["jpg", "txt"]
        if col in dataset.column_names
    ]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)

    if not isinstance(dataset, NestedDataset):
        dataset = NestedDataset(dataset)
    return dataset


def preprocess_danqing(dataset, num_proc=1):
    """
    Preprocess DanQing (HuggingFace image+caption parquet) format dataset.

    DanQing shards carry columns 'images' (struct<bytes>), 'alt_text' and
    'recaption'. This reshapes them into the format expected by Data-Juicer:
    - Flattens the 'images' struct<bytes> into 'image_bytes' (raw JPEG bytes)
      and creates 'images' (synthetic one-element ID list per row); the bytes
      are the real payload the ops consume.
    - Builds 'text' from 'alt_text', prepended with the image special token so
      multimodal ops can correlate the image with text. 'alt_text' is dropped;
      'recaption' is kept as its own passthrough column.

    :param dataset: a NestedDataset with DanQing schema
    :param num_proc: number of processes for mapping
    :return: transformed NestedDataset
    """
    import io

    from data_juicer.core.data import NestedDataset
    from data_juicer.utils.mm_utils import SpecialTokens

    logger.info("Applying DanQing (image+caption parquet) preprocessing...")

    def transform(sample, idx):
        # Extract raw image bytes; datasets may surface 'images' as a struct
        # dict ({'bytes': ...}) or as a decoded PIL Image.
        img = sample.get("images")
        raw = None
        if isinstance(img, dict):
            raw = img.get("bytes")
        elif img is not None and hasattr(img, "save"):
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            raw = buf.getvalue()
        elif isinstance(img, (bytes, bytearray)):
            raw = bytes(img)

        if raw is not None:
            sample["image_bytes"] = [raw]
            sample["images"] = [f"{idx}.jpg"]
        else:
            sample["image_bytes"] = []
            sample["images"] = []

        # Build text from alt_text, prefixed with the image special token
        alt = sample.get("alt_text", "") or ""
        sample["text"] = SpecialTokens.image + alt

        return sample

    dataset = dataset.map(
        transform, with_indices=True, num_proc=num_proc,
        desc="DanQing preprocessing"
    )

    # Remove the consumed caption column ('images' is overwritten above;
    # 'recaption' is kept as a passthrough column)
    cols_to_remove = [
        col
        for col in ["alt_text"]
        if col in dataset.column_names
    ]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)

    if not isinstance(dataset, NestedDataset):
        dataset = NestedDataset(dataset)
    return dataset


def unify_format(
    dataset: Dataset,
    text_keys: Union[List[str], str] = "text",
    num_proc: int = 1,
    global_cfg: Union[dict, Namespace] = None,
) -> Dataset:
    """
    Get an unified internal format, conduct the following modifications.

    1. check keys of dataset

    2. filter out those samples with empty or None text

    :param dataset: input dataset
    :param text_keys: original text key(s) of dataset.
    :param num_proc: number of processes for mapping
    :param global_cfg: the global cfg used in consequent processes,
        since cfg.text_key may be modified after unifying

    :return: unified_format_dataset
    """
    from data_juicer.core.data import NestedDataset

    if isinstance(dataset, DatasetDict):
        datasets = list(dataset.values())
        assert len(datasets) == 1, "Please make sure the passed datasets " "contains only 1 dataset"
        dataset = datasets[0]
    assert isinstance(dataset, Dataset) or isinstance(dataset, NestedDataset), (
        "Currently we only support processing data" "with huggingface-Dataset format"
    )

    if text_keys is None:
        text_keys = []

    if isinstance(text_keys, str):
        text_keys = [text_keys]

    logger.info("Unifying the input dataset formats...")

    dataset = NestedDataset(dataset)

    # 1. check text related keys
    for key in text_keys:
        if key not in dataset.features:
            err_msg = (
                f"There is no key [{key}] in dataset. You might set "
                f"wrong text_key in the config file for your dataset. "
                f"Please check and retry!"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

    # 2. filter out those samples with empty or None text
    # TODO: optimize the filtering operation for better efficiency
    logger.info(f"There are {len(dataset)} sample(s) in the original dataset.")

    def non_empty_text(sample, target_keys):
        for target_key in target_keys:
            # TODO: case for CFT, in which the len(sample[target_key]) == 0
            if sample[target_key] is None:
                # we filter out the samples contains at least None column
                # since the op can not handle it now
                return False
        return True

    dataset = dataset.filter(non_empty_text, num_proc=num_proc, fn_kwargs={"target_keys": text_keys})
    logger.info(f"{len(dataset)} samples left after filtering empty text.")

    # 3. convert relative paths to absolute paths
    if global_cfg is not None:
        if isinstance(global_cfg, dict):
            global_cfg = dict_to_namespace(global_cfg)
        # check and get dataset dir
        if (
            hasattr(global_cfg, "dataset_path")
            and global_cfg.dataset_path is not None
            and os.path.exists(global_cfg.dataset_path)
        ):
            if os.path.isdir(global_cfg.dataset_path):
                ds_dir = global_cfg.dataset_path
            else:
                ds_dir = os.path.dirname(global_cfg.dataset_path)
        else:
            ds_dir = ""
        image_key = global_cfg.image_key if hasattr(global_cfg, "image_key") else "images"
        audio_key = global_cfg.audio_key if hasattr(global_cfg, "audio_key") else "audios"
        video_key = global_cfg.video_key if hasattr(global_cfg, "video_key") else "videos"

        data_path_keys = []
        if image_key in dataset.features:
            data_path_keys.append(image_key)
        if audio_key in dataset.features:
            data_path_keys.append(audio_key)
        if video_key in dataset.features:
            data_path_keys.append(video_key)
        if len(data_path_keys) == 0:
            # no image/audio/video path list in dataset, no need to convert
            return dataset

        if ds_dir == "":
            return dataset

        logger.info(
            "Converting relative paths in the dataset to their "
            "absolute version. (Based on the directory of input "
            "dataset file)"
        )

        # function to convert relative paths to absolute paths
        def rel2abs(sample, path_keys, dataset_dir):
            for path_key in path_keys:
                if path_key not in sample:
                    continue
                paths = sample[path_key]
                if not paths:
                    continue
                new_paths = [path if is_absolute_path(path) else os.path.join(dataset_dir, path) for path in paths]
                sample[path_key] = new_paths
            return sample

        dataset = dataset.map(
            rel2abs, num_proc=num_proc, fn_kwargs={"path_keys": data_path_keys, "dataset_dir": ds_dir}
        )
    else:
        logger.warning(
            "No global config passed into unify_format function. "
            "Relative paths in the dataset might not be converted "
            "to their absolute versions. Data of other modalities "
            "might not be able to find by Data-Juicer."
        )

    return dataset
