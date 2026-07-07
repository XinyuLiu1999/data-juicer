from __future__ import annotations

import os
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import pyarrow
import ray
from jsonargparse import Namespace
from loguru import logger
from ray.data._internal.util import get_compute_strategy

from data_juicer.core.data import DJDataset
from data_juicer.core.data.schema import Schema
from data_juicer.core.tracer import should_trace_op
from data_juicer.ops import Deduplicator, Filter, Mapper, Pipeline
from data_juicer.ops.base_op import DEFAULT_BATCH_SIZE, TAGGING_OPS
from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import is_remote_path
from data_juicer.utils.webdataset_utils import _custom_default_decoder


def get_abs_path(path, dataset_dir):
    if is_remote_path(path):
        return path
    path = os.path.join(dataset_dir, path)
    if is_remote_path(path):
        return path
    full_path = os.path.abspath(path)
    if os.path.exists(full_path):
        return full_path
    else:
        return path


def convert_to_absolute_paths(samples: pyarrow.Table, dataset_dir, path_keys):
    for key in path_keys:
        col_idx = samples.schema.get_field_index(key)
        cols = samples.column(col_idx)

        def _process_paths():
            for col in cols:
                path = col.as_py()
                if isinstance(path, str):
                    yield get_abs_path(path, dataset_dir)
                elif isinstance(path, list):
                    yield [get_abs_path(p, dataset_dir) for p in path]
                else:
                    yield path

        samples = samples.set_column(col_idx, key, pyarrow.array(_process_paths()))
    return samples


# TODO: check path for nestdataset
def set_dataset_to_absolute_path(dataset, dataset_path, cfg, columns=None):
    """
    Set all the path in input data to absolute path.
    Checks dataset_dir and project_dir for valid paths.
    """
    path_keys = []
    if columns is None:
        columns = dataset.columns()
    for key in [
        cfg.get("video_key", "videos"),
        cfg.get("image_key", "images"),
        cfg.get("audio_key", "audios"),
    ]:
        if key in columns:
            path_keys.append(key)
    if len(path_keys) > 0:
        dataset_dir = os.path.dirname(dataset_path)
        logger.info(f"dataset_dir: {dataset_dir}")
        dataset = dataset.map_batches(
            partial(convert_to_absolute_paths, dataset_dir=dataset_dir, path_keys=path_keys),
            batch_format="pyarrow",
            zero_copy_batch=True,
            batch_size=DEFAULT_BATCH_SIZE,
        )
    return dataset


def _safe_extract_text(raw, cnt, image_special_token, _loads):
    """Extract text from a clean_content JSON string."""
    if raw is None:
        return ""
    try:
        return image_special_token * cnt + _loads(raw).get("text", "")
    except Exception:
        return ""


def preprocess_laioncoco_ray(table: pyarrow.Table, image_special_token: str) -> pyarrow.Table:
    """
    Ray-compatible LAION-COCO preprocessing operating on PyArrow tables.

    Transforms the non-standard LAION-COCO schema into the format
    expected by Data-Juicer:
    - Extracts 'text' from the 'clean_content' JSON string, prepended
      with image special tokens
    - Flattens 'image_buffer_list' structs into 'images' (IDs) and
      'image_bytes' (raw bytes) columns

    Uses per-chunk PyArrow operations to avoid copying image bytes,
    which is the main performance bottleneck for large tables.
    """
    try:
        import orjson
        _loads = orjson.loads
    except ImportError:
        import json
        _loads = json.loads

    import pyarrow.compute as pc

    num_rows = table.num_rows

    # --- Extract image_ids and image_bytes via per-chunk struct access ---
    # Process each chunk independently to avoid combine_chunks(), which
    # copies all image bytes into a contiguous buffer. Instead, we flatten
    # each chunk separately and build chunked arrays (zero-copy).
    if "image_buffer_list" in table.column_names:
        buf_chunked = table.column("image_buffer_list")
        id_chunks = []
        byte_chunks = []
        img_count_chunks = []

        for chunk in buf_chunked.chunks:
            offsets = chunk.offsets
            flattened = chunk.flatten()

            if len(flattened) > 0 and flattened.type.num_fields > 0:
                field_names = [flattened.type.field(i).name
                               for i in range(flattened.type.num_fields)]
                ids = (flattened.field("image_id")
                       if "image_id" in field_names
                       else pyarrow.nulls(len(flattened),
                                          type=pyarrow.string()))
                bts = (flattened.field("buffer")
                       if "buffer" in field_names
                       else pyarrow.nulls(len(flattened),
                                          type=pyarrow.binary()))
            else:
                ids = pyarrow.array([], type=pyarrow.string())
                bts = pyarrow.array([], type=pyarrow.binary())

            # Zero-base offsets: sliced chunks (from Ray batching) may
            # have offsets that don't start at 0 (e.g. [3, 4, 5]),
            # while flatten() returns only the values for this slice.
            # ListArray.from_arrays requires offsets starting at 0.
            first_offset = offsets[0].as_py()
            if first_offset != 0:
                offsets = pc.subtract(offsets, first_offset)

            id_chunks.append(
                pyarrow.ListArray.from_arrays(offsets, ids))
            byte_chunks.append(
                pyarrow.ListArray.from_arrays(offsets, bts))
            img_count_chunks.append(
                pc.subtract(offsets[1:].cast(pyarrow.int64()),
                            offsets[:-1].cast(pyarrow.int64())))

        images_col = pyarrow.chunked_array(id_chunks)
        image_bytes_col = pyarrow.chunked_array(byte_chunks)
        img_counts = pyarrow.chunked_array(img_count_chunks)
    else:
        empty_str = pyarrow.array([], type=pyarrow.string())
        empty_bin = pyarrow.array([], type=pyarrow.binary())
        zero_offsets = pyarrow.array([0] * (num_rows + 1),
                                     type=pyarrow.int32())
        images_col = pyarrow.chunked_array(
            [pyarrow.ListArray.from_arrays(zero_offsets, empty_str)])
        image_bytes_col = pyarrow.chunked_array(
            [pyarrow.ListArray.from_arrays(zero_offsets, empty_bin)])
        img_counts = pyarrow.chunked_array(
            [pyarrow.array([0] * num_rows, type=pyarrow.int64())])

    # --- Extract text from clean_content JSON ---
    clean_content_list = table.column("clean_content").to_pylist()
    img_counts_list = img_counts.to_pylist()
    text_col = [
        _safe_extract_text(raw, cnt, image_special_token, _loads)
        for raw, cnt in zip(clean_content_list, img_counts_list)
    ]

    # --- Build output table, dropping original columns ---
    cols_to_keep = [name for name in table.column_names
                    if name not in ("clean_content", "image_buffer_list")]
    out_table = table.select(cols_to_keep)
    out_table = out_table.append_column("text", pyarrow.array(text_col))
    out_table = out_table.append_column("images", images_col)
    out_table = out_table.append_column("image_bytes", image_bytes_col)

    return out_table


# ---------------------------------------------------------------------------
# Unified OCR preprocessing
#
# One preprocessor for a family of heterogeneous OCR / grounding parquet
# datasets (c4web, docmatix, blip3, ocr_0926, wukong, zhihu_box/pure,
# laion_hehe, ...). It auto-detects the content and image columns per source
# and normalises them into the standard Data-Juicer format (``text`` /
# ``images`` / ``image_bytes``), so a single config works for every source
# (one DJ run per source; a single Ray dataset must have one schema).
#
# Column-name candidates mirror collect_unified_ocr.py.
# ---------------------------------------------------------------------------
UNIFIED_CONTENT_COLS = ["clean_content", "origin_content", "0"]
UNIFIED_IMG_LIST_COLS = ["image_buffer_list", "1"]   # list<struct{...}>
UNIFIED_IMG_BIN_COLS = ["img_bytes", "img_byte"]     # flat binary (one/row)
UNIFIED_IMG_BYTE_FIELDS = ["buffer", "image_bytes", "img_bytes"]  # in struct


def detect_unified_ocr_columns(columns):
    """Detect (content_col, image_col, image_type) from a set of column names.

    ``image_type`` is ``"list"`` for list-of-struct image columns
    (``image_buffer_list`` / ``"1"``) or ``"binary"`` for flat binary columns
    (``img_bytes`` / ``img_byte``). Any of the three may be ``None`` when not
    found; callers decide whether that means "skip" or "error".
    """
    content_col = next((c for c in UNIFIED_CONTENT_COLS if c in columns), None)
    for c in UNIFIED_IMG_LIST_COLS:
        if c in columns:
            return content_col, c, "list"
    for c in UNIFIED_IMG_BIN_COLS:
        if c in columns:
            return content_col, c, "binary"
    return content_col, None, None


def _pick_struct_field(flattened, field_names, candidates, arrow_type):
    """Return the first struct field in ``candidates`` present on the flattened
    struct array, or an all-null array of ``arrow_type`` if none match."""
    for name in candidates:
        if name in field_names:
            return flattened.field(name)
    return pyarrow.nulls(len(flattened), type=arrow_type)


def _synth_ids(n):
    """Synthesise ``n`` positional image IDs. IDs are only used downstream as
    per-sample dict keys (the raw bytes are the real payload), so positional
    values guarantee uniqueness within a sample regardless of source schema."""
    return pyarrow.array([f"{k}.jpg" for k in range(n)], type=pyarrow.string())


def _flatten_struct_images(buf_chunked):
    """Flatten a list<struct{buffer/image_id/...}> column into
    ``(images, image_bytes, img_counts)`` chunked arrays, zero-copy per chunk
    (mirrors preprocess_laioncoco_ray). Only the *bytes* field name is
    generalised across sources; IDs are synthesised positionally."""
    import pyarrow.compute as pc

    id_chunks, byte_chunks, count_chunks = [], [], []
    for chunk in buf_chunked.chunks:
        offsets = chunk.offsets
        flattened = chunk.flatten()

        if len(flattened) > 0 and flattened.type.num_fields > 0:
            field_names = [flattened.type.field(i).name
                           for i in range(flattened.type.num_fields)]
            bts = _pick_struct_field(flattened, field_names,
                                     UNIFIED_IMG_BYTE_FIELDS, pyarrow.binary())
        else:
            bts = pyarrow.array([], type=pyarrow.binary())

        # Zero-base offsets: sliced chunks (from Ray batching) may start at a
        # non-zero offset while flatten() returns only this slice's values.
        first_offset = offsets[0].as_py()
        if first_offset != 0:
            offsets = pc.subtract(offsets, first_offset)

        byte_chunks.append(pyarrow.ListArray.from_arrays(offsets, bts))
        id_chunks.append(
            pyarrow.ListArray.from_arrays(offsets, _synth_ids(len(bts))))
        count_chunks.append(
            pc.subtract(offsets[1:].cast(pyarrow.int64()),
                        offsets[:-1].cast(pyarrow.int64())))

    return (pyarrow.chunked_array(id_chunks),
            pyarrow.chunked_array(byte_chunks),
            pyarrow.chunked_array(count_chunks))


def _flatten_binary_images(bin_chunked):
    """Wrap a flat binary image column (one image per row) into
    ``(images, image_bytes, img_counts)`` chunked arrays. Null rows become
    empty lists (0 images, 0 tokens) so token count stays aligned with the
    number of images actually present."""
    id_chunks, byte_chunks, count_chunks = [], [], []
    for chunk in bin_chunked.chunks:
        valid = chunk.is_valid().to_pylist()
        offsets_list = [0]
        for v in valid:
            offsets_list.append(offsets_list[-1] + (1 if v else 0))
        offsets = pyarrow.array(offsets_list, type=pyarrow.int32())

        values = chunk.filter(chunk.is_valid())  # drop null rows' payload
        byte_chunks.append(pyarrow.ListArray.from_arrays(offsets, values))
        id_chunks.append(
            pyarrow.ListArray.from_arrays(offsets, _synth_ids(len(values))))
        count_chunks.append(
            pyarrow.array([1 if v else 0 for v in valid], type=pyarrow.int64()))

    return (pyarrow.chunked_array(id_chunks),
            pyarrow.chunked_array(byte_chunks),
            pyarrow.chunked_array(count_chunks))


def preprocess_unified_ocr_ray(
    table: pyarrow.Table,
    image_special_token: str,
    content_col: str,
    image_col: str,
    image_type: str,
) -> pyarrow.Table:
    """
    Ray-compatible unified OCR preprocessing operating on PyArrow tables.

    Normalises a heterogeneous OCR/grounding parquet schema into the format
    expected by Data-Juicer:
    - Extracts ``text`` from the ``content_col`` JSON string (key ``"text"``),
      prepended with one image special token per image in the row.
    - Flattens ``image_col`` into ``images`` (synthetic IDs) and ``image_bytes``
      (raw bytes). Both list-of-struct and flat-binary image columns are
      supported (``image_type`` = ``"list"`` / ``"binary"``).

    ``content_col`` / ``image_col`` / ``image_type`` are resolved once by the
    caller (see ``detect_unified_ocr_columns``) since a Ray dataset has a single
    schema, avoiding per-batch re-detection.
    """
    try:
        import orjson
        _loads = orjson.loads
    except ImportError:
        import json
        _loads = json.loads

    if image_type == "binary":
        images_col, image_bytes_col, img_counts = _flatten_binary_images(
            table.column(image_col))
    else:
        images_col, image_bytes_col, img_counts = _flatten_struct_images(
            table.column(image_col))

    # --- Extract text from the content JSON (orjson accepts str and bytes) ---
    content_list = table.column(content_col).to_pylist()
    img_counts_list = img_counts.to_pylist()
    text_col = [
        _safe_extract_text(raw, cnt, image_special_token, _loads)
        for raw, cnt in zip(content_list, img_counts_list)
    ]

    # --- Build output table, dropping the consumed content + image columns.
    #     Any other columns (e.g. uid, clip_score) pass through untouched. ---
    cols_to_keep = [name for name in table.column_names
                    if name not in (content_col, image_col)]
    out_table = table.select(cols_to_keep)
    out_table = out_table.append_column("text", pyarrow.array(text_col))
    out_table = out_table.append_column("images", images_col)
    out_table = out_table.append_column("image_bytes", image_bytes_col)

    return out_table


def preprocess_blip3o_ray(table: pyarrow.Table, image_special_token: str) -> pyarrow.Table:
    """
    Ray-compatible BLIP3o WebDataset preprocessing operating on PyArrow tables.

    Transforms the BLIP3o WebDataset schema (columns: jpg, txt, __key__,
    __url__) into the standard Data-Juicer format:
    - Renames 'txt' to 'text', prepended with an image special token
    - Wraps 'jpg' (raw JPEG bytes) into 'image_bytes' (list of bytes) and
      creates 'images' (synthetic IDs derived from __key__)

    Expects images as raw bytes (not PIL), which is the output of
    _blip3o_decoder.
    """
    num_rows = table.num_rows

    # --- Wrap raw image bytes from 'jpg' column ---
    jpg_col = table.column("jpg").to_pylist()
    key_col = table.column("__key__").to_pylist()

    image_ids = []
    image_bytes_list = []
    for i in range(num_rows):
        img_bytes = jpg_col[i]
        key = key_col[i] if key_col[i] is not None else str(i)
        if img_bytes is not None:
            image_bytes_list.append(img_bytes)
            image_ids.append(f"{key}.jpg")
        else:
            image_bytes_list.append(None)
            image_ids.append(None)

    # Wrap each value in a single-element list (Data-Juicer expects lists)
    images_col = pyarrow.array(
        [[iid] if iid is not None else [] for iid in image_ids],
        type=pyarrow.list_(pyarrow.string()),
    )
    image_bytes_col = pyarrow.array(
        [[ib] if ib is not None else [] for ib in image_bytes_list],
        type=pyarrow.list_(pyarrow.binary()),
    )

    # --- Build text column from 'txt', prepend image special token ---
    txt_col = table.column("txt").to_pylist()
    text_col = pyarrow.array([
        (image_special_token + (t if t is not None else ""))
        for t in txt_col
    ])

    # --- Build output table, dropping original webdataset columns ---
    cols_to_drop = {"jpg", "txt"}
    cols_to_keep = [name for name in table.column_names
                    if name not in cols_to_drop]
    out_table = table.select(cols_to_keep)
    out_table = out_table.append_column("text", text_col)
    out_table = out_table.append_column("images", images_col)
    out_table = out_table.append_column("image_bytes", image_bytes_col)

    return out_table


def preprocess_taisu_ray(table: pyarrow.Table, image_special_token: str) -> pyarrow.Table:
    """
    Ray-compatible TaiSu (image-only WebDataset) preprocessing on PyArrow tables.

    TaiSu tars contain only images (a 'jpg' column, no 'txt'/'json' caption).
    This mirrors preprocess_blip3o_ray but does NOT require a 'txt' column:
    - Wraps 'jpg' (raw JPEG bytes) into 'image_bytes' (list of bytes) and
      creates 'images' (synthetic IDs derived from __key__)
    - Synthesizes 'text' as just the image special token (empty caption), so
      multimodal ops (e.g. image_text_similarity_filter) still form an
      image-text chunk and can emit the CLIP image embedding. If a 'txt' column
      happens to be present its value is appended after the token.

    Expects images as raw bytes (not PIL), i.e. the output of _blip3o_decoder.
    """
    num_rows = table.num_rows
    column_names = set(table.column_names)

    jpg_col = table.column("jpg").to_pylist()
    key_col = (table.column("__key__").to_pylist()
               if "__key__" in column_names else [None] * num_rows)
    txt_col = (table.column("txt").to_pylist()
               if "txt" in column_names else [None] * num_rows)

    image_ids = []
    image_bytes_list = []
    for i in range(num_rows):
        img_bytes = jpg_col[i]
        key = key_col[i] if key_col[i] is not None else str(i)
        if img_bytes is not None:
            image_bytes_list.append(img_bytes)
            image_ids.append(f"{key}.jpg")
        else:
            image_bytes_list.append(None)
            image_ids.append(None)

    images_col = pyarrow.array(
        [[iid] if iid is not None else [] for iid in image_ids],
        type=pyarrow.list_(pyarrow.string()),
    )
    image_bytes_col = pyarrow.array(
        [[ib] if ib is not None else [] for ib in image_bytes_list],
        type=pyarrow.list_(pyarrow.binary()),
    )
    # No caption in TaiSu: text is just the image special token (empty caption).
    text_col = pyarrow.array([
        image_special_token + (txt_col[i] if txt_col[i] is not None else "")
        for i in range(num_rows)
    ])

    cols_to_drop = {"jpg", "txt"}
    cols_to_keep = [name for name in table.column_names
                    if name not in cols_to_drop]
    out_table = table.select(cols_to_keep)
    out_table = out_table.append_column("text", text_col)
    out_table = out_table.append_column("images", images_col)
    out_table = out_table.append_column("image_bytes", image_bytes_col)

    return out_table


def preprocess_danqing_ray(table: pyarrow.Table, image_special_token: str) -> pyarrow.Table:
    """
    Ray-compatible DanQing preprocessing operating on PyArrow tables.

    DanQing shards are HuggingFace image+caption parquet with columns
    'images' (struct<bytes>), 'alt_text' and 'recaption'. This reshapes them
    into Data-Juicer's standard multimodal schema:
    - Flattens the 'images' struct<bytes> column into 'image_bytes' (list of
      raw bytes) and synthesizes 'images' (a one-element list of string IDs
      per row; the bytes are the real payload the ops consume).
    - Builds 'text' from the 'alt_text' caption, prepended with the image
      special token so multimodal ops (e.g. image_text_similarity_filter)
      form an image-text chunk. 'alt_text' is dropped; 'recaption' is kept as
      its own passthrough column.

    Each row holds exactly one image, so per-chunk unit-length list wrapping
    keeps image bytes zero-copy (no combine_chunks(), which would copy every
    image payload into one contiguous buffer).
    """
    num_rows = table.num_rows

    # --- Flatten images struct<bytes> into image_bytes (one image per row) ---
    if "images" in table.column_names:
        img_chunked = table.column("images")
        byte_chunks = []
        for chunk in img_chunked.chunks:
            n = len(chunk)
            field_names = [chunk.type.field(i).name
                           for i in range(chunk.type.num_fields)]
            bts = (chunk.field("bytes")
                   if "bytes" in field_names
                   else pyarrow.nulls(n, type=pyarrow.binary()))
            # one image per row -> unit-length lists, offsets [0, 1, ..., n]
            offsets = pyarrow.array(range(n + 1), type=pyarrow.int32())
            byte_chunks.append(
                pyarrow.ListArray.from_arrays(offsets, bts))
        image_bytes_col = pyarrow.chunked_array(
            byte_chunks, type=pyarrow.list_(pyarrow.binary()))
    else:
        empty_bin = pyarrow.array([], type=pyarrow.binary())
        zero_offsets = pyarrow.array([0] * (num_rows + 1),
                                     type=pyarrow.int32())
        image_bytes_col = pyarrow.chunked_array(
            [pyarrow.ListArray.from_arrays(zero_offsets, empty_bin)])

    # --- Synthesize one image ID per row (bytes are the real payload) ---
    images_col = pyarrow.array(
        [[f"{i}.jpg"] for i in range(num_rows)],
        type=pyarrow.list_(pyarrow.string()),
    )

    # --- Build text from alt_text, prefixed with the image special token ---
    alt_col = (table.column("alt_text").to_pylist()
               if "alt_text" in table.column_names else [None] * num_rows)
    text_col = pyarrow.array([
        image_special_token + (alt_col[i] if alt_col[i] is not None else "")
        for i in range(num_rows)
    ])

    # --- Build output table, dropping the raw image struct + consumed caption
    #     ('recaption' is kept as a passthrough column) ---
    cols_to_drop = {"images", "alt_text"}
    cols_to_keep = [name for name in table.column_names
                    if name not in cols_to_drop]
    out_table = table.select(cols_to_keep)
    out_table = out_table.append_column("text", text_col)
    out_table = out_table.append_column("images", images_col)
    out_table = out_table.append_column("image_bytes", image_bytes_col)

    return out_table


def preprocess_dataset(dataset: ray.data.Dataset, dataset_path, cfg):
    """Preprocess dataset and return (dataset, cached_columns).

    Returns the columns set so callers can avoid extra .columns() calls
    on the lazy dataset, which can trigger Ray Data internal queue
    assertion errors.
    """
    # Get columns once on the raw dataset before chaining lazy ops
    columns = set(dataset.columns())

    if dataset_path:
        dataset = set_dataset_to_absolute_path(
            dataset, dataset_path, cfg, columns=columns
        )

    if cfg and getattr(cfg, "laioncoco_preprocessing", False):
        # Skip if preprocessing was already applied (clean_content removed)
        if "clean_content" in columns:
            from data_juicer.utils.mm_utils import SpecialTokens

            preprocessing_num_cpus = getattr(
                cfg, "laioncoco_preprocessing_num_cpus", 0.25
            )
            preprocessing_batch_size = getattr(
                cfg, "laioncoco_preprocessing_batch_size",
                DEFAULT_BATCH_SIZE
            )
            logger.info(
                f"Applying LAION-COCO preprocessing for Ray dataset "
                f"(num_cpus={preprocessing_num_cpus}, "
                f"batch_size={preprocessing_batch_size})..."
            )
            dataset = dataset.map_batches(
                partial(preprocess_laioncoco_ray,
                        image_special_token=SpecialTokens.image),
                batch_format="pyarrow",
                batch_size=preprocessing_batch_size,
                num_cpus=preprocessing_num_cpus,
            )
            # Update columns to reflect laioncoco preprocessing changes
            columns.discard("clean_content")
            columns.discard("image_buffer_list")
            columns.update(["text", "images", "image_bytes"])
        else:
            logger.info("LAION-COCO preprocessing already applied, skipping.")

    if cfg and getattr(cfg, "unified_ocr_preprocessing", False):
        content_col, image_col, image_type = detect_unified_ocr_columns(columns)
        # Skip if already preprocessed (text + image_bytes already present).
        already_applied = "text" in columns and "image_bytes" in columns
        if content_col and image_col and not already_applied:
            from data_juicer.utils.mm_utils import SpecialTokens

            preprocessing_num_cpus = getattr(
                cfg, "unified_ocr_preprocessing_num_cpus", 0.25
            )
            preprocessing_batch_size = getattr(
                cfg, "unified_ocr_preprocessing_batch_size",
                DEFAULT_BATCH_SIZE
            )
            logger.info(
                f"Applying unified OCR preprocessing for Ray dataset "
                f"(content='{content_col}', image='{image_col}' ({image_type}), "
                f"num_cpus={preprocessing_num_cpus}, "
                f"batch_size={preprocessing_batch_size})..."
            )
            dataset = dataset.map_batches(
                partial(preprocess_unified_ocr_ray,
                        image_special_token=SpecialTokens.image,
                        content_col=content_col,
                        image_col=image_col,
                        image_type=image_type),
                batch_format="pyarrow",
                batch_size=preprocessing_batch_size,
                num_cpus=preprocessing_num_cpus,
            )
            # Update columns to reflect unified OCR preprocessing changes
            columns.discard(content_col)
            columns.discard(image_col)
            columns.update(["text", "images", "image_bytes"])
        elif already_applied:
            logger.info("Unified OCR preprocessing already applied, skipping.")
        else:
            logger.warning(
                f"Unified OCR preprocessing enabled but could not detect "
                f"columns (content={content_col}, image={image_col}); "
                f"available columns: {sorted(columns)}. Skipping."
            )

    if cfg and getattr(cfg, "blip3o_preprocessing", False):
        # Skip if preprocessing was already applied (jpg column removed)
        if "jpg" in columns:
            from data_juicer.utils.mm_utils import SpecialTokens

            preprocessing_num_cpus = getattr(
                cfg, "blip3o_preprocessing_num_cpus", 0.25
            )
            preprocessing_batch_size = getattr(
                cfg, "blip3o_preprocessing_batch_size",
                DEFAULT_BATCH_SIZE
            )
            logger.info(
                f"Applying BLIP3o WebDataset preprocessing for Ray dataset "
                f"(num_cpus={preprocessing_num_cpus}, "
                f"batch_size={preprocessing_batch_size})..."
            )
            dataset = dataset.map_batches(
                partial(preprocess_blip3o_ray,
                        image_special_token=SpecialTokens.image),
                batch_format="pyarrow",
                batch_size=preprocessing_batch_size,
                num_cpus=preprocessing_num_cpus,
            )
            # Update columns to reflect blip3o preprocessing changes
            columns.discard("jpg")
            columns.discard("txt")
            columns.update(["text", "images", "image_bytes"])
        else:
            logger.info("BLIP3o preprocessing already applied, skipping.")

    if cfg and getattr(cfg, "taisu_preprocessing", False):
        # Skip if preprocessing was already applied (jpg column removed)
        if "jpg" in columns:
            from data_juicer.utils.mm_utils import SpecialTokens

            preprocessing_num_cpus = getattr(
                cfg, "taisu_preprocessing_num_cpus", 0.25
            )
            preprocessing_batch_size = getattr(
                cfg, "taisu_preprocessing_batch_size",
                DEFAULT_BATCH_SIZE
            )
            logger.info(
                f"Applying TaiSu (image-only WebDataset) preprocessing for Ray "
                f"dataset (num_cpus={preprocessing_num_cpus}, "
                f"batch_size={preprocessing_batch_size})..."
            )
            dataset = dataset.map_batches(
                partial(preprocess_taisu_ray,
                        image_special_token=SpecialTokens.image),
                batch_format="pyarrow",
                batch_size=preprocessing_batch_size,
                num_cpus=preprocessing_num_cpus,
            )
            # Update columns to reflect taisu preprocessing changes
            columns.discard("jpg")
            columns.discard("txt")
            columns.update(["text", "images", "image_bytes"])
        else:
            logger.info("TaiSu preprocessing already applied, skipping.")

    if cfg and getattr(cfg, "danqing_preprocessing", False):
        # Skip if preprocessing was already applied (alt_text removed)
        if "alt_text" in columns:
            from data_juicer.utils.mm_utils import SpecialTokens

            preprocessing_num_cpus = getattr(
                cfg, "danqing_preprocessing_num_cpus", 0.25
            )
            preprocessing_batch_size = getattr(
                cfg, "danqing_preprocessing_batch_size",
                DEFAULT_BATCH_SIZE
            )
            logger.info(
                f"Applying DanQing (image+caption parquet) preprocessing for "
                f"Ray dataset (num_cpus={preprocessing_num_cpus}, "
                f"batch_size={preprocessing_batch_size})..."
            )
            dataset = dataset.map_batches(
                partial(preprocess_danqing_ray,
                        image_special_token=SpecialTokens.image),
                batch_format="pyarrow",
                batch_size=preprocessing_batch_size,
                num_cpus=preprocessing_num_cpus,
            )
            # Update columns to reflect danqing preprocessing changes
            # ('recaption' is kept as a passthrough column)
            columns.discard("alt_text")
            columns.update(["text", "images", "image_bytes"])
        else:
            logger.info("DanQing preprocessing already applied, skipping.")

    return dataset, columns


def filter_batch(batch, filter_func):
    mask = pyarrow.array(filter_func(batch.to_pydict()))
    return batch.filter(mask)


def _is_valid_parquet(path):
    """True if ``path`` has a readable Parquet footer (metadata)."""
    try:
        import pyarrow.parquet as pq

        pq.ParquetFile(path).metadata
        return True
    except Exception as e:
        logger.warning(f"Skipping unreadable parquet shard {path}: {e}")
        return False


def filter_valid_parquet(paths):
    """Expand ``paths`` (a local dir, file, or list) into individual parquet
    files and drop any with a corrupt/truncated footer, validating in parallel.

    Returns a list of good file paths. Remote paths are returned unchanged (the
    footer check only works on the local filesystem). Raises if every shard is
    unreadable, since that almost certainly signals a path/permissions problem
    rather than universal corruption.
    """
    import glob
    from concurrent.futures import ThreadPoolExecutor

    raw = paths if isinstance(paths, (list, tuple)) else [paths]
    if any(is_remote_path(p) for p in raw):
        return paths

    files = []
    for p in raw:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True)))
        else:
            files.append(p)
    files = [f for f in files if not os.path.basename(f).startswith("_")]
    if not files:
        return paths  # nothing to validate; let ray.data surface the error

    with ThreadPoolExecutor(max_workers=32) as ex:
        valid_flags = list(ex.map(_is_valid_parquet, files))
    good = [f for f, ok in zip(files, valid_flags) if ok]
    skipped = len(files) - len(good)

    if not good:
        raise RuntimeError(
            f"All {len(files)} parquet shards under {raw} failed the footer "
            f"check. This usually means a wrong path or permissions issue, "
            f"not universal corruption."
        )
    if skipped:
        logger.warning(
            f"Skipping {skipped}/{len(files)} corrupt parquet shard(s); "
            f"reading the remaining {len(good)}."
        )
    return good


class RayDataset(DJDataset):
    def __init__(
        self,
        dataset: ray.data.Dataset,
        dataset_path: str = None,
        cfg: Optional[Namespace] = None,
        auto_op_parallelism=True,
    ) -> None:
        self.data, self._cached_columns = preprocess_dataset(
            dataset, dataset_path, cfg
        )

        # if auto_op_parallelism is set in both args and cfg, cfg takes precedence
        if cfg and cfg.get("auto_op_parallelism") is not None:
            self._auto_proc = cfg.get("auto_op_parallelism")
        else:
            self._auto_proc = auto_op_parallelism

    def schema(self) -> Schema:
        """Get dataset schema.

        Returns:
            Schema: Dataset schema containing column names and types
        """
        if self.data is None or not self._cached_columns:
            raise ValueError("Dataset is empty or not initialized")

        return Schema.from_ray_schema(self.data.schema())

    def get(self, k: int) -> List[Dict[str, Any]]:
        """Get k rows from the dataset."""
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")

        if k == 0:
            return []

        k = min(k, self.data.count())
        return list(self.data.limit(k).take())

    def get_column(self, column: str, k: Optional[int] = None) -> List[Any]:
        """Get column values from Ray dataset.

        Args:
            column: Name of the column to retrieve
            k: Optional number of rows to return. If None, returns all rows

        Returns:
            List of values from the specified column

        Raises:
            KeyError: If column doesn't exist
            ValueError: If k is negative
        """
        if self.data is None or column not in self._cached_columns:
            raise KeyError(f"Column '{column}' not found in dataset")

        if k is not None:
            if k < 0:
                raise ValueError(f"k must be non-negative, got {k}")
            if k == 0:
                return []
            k = min(k, self.data.count())
            return [row[column] for row in self.data.limit(k).take()]

        return [row[column] for row in self.data.take()]

    def process(self, operators, *, exporter=None, checkpointer=None, tracer=None) -> DJDataset:
        if operators is None:
            return self
        if not isinstance(operators, list):
            operators = [operators]

        from data_juicer.utils.process_utils import calculate_ray_np

        if self._auto_proc:
            calculate_ray_np(operators)

        # Use cached columns from initialization to avoid calling
        # self.data.columns() which can trigger Ray Data internal queue
        # assertion errors on datasets with pending lazy operations.
        cached_columns = set(self._cached_columns)

        for op in operators:
            try:
                cached_columns = self._run_single_op(op, cached_columns, tracer=tracer)
            except Exception as e:
                logger.error(f"Error processing operator {op}: {e}.")
                if getattr(op, '_user_runtime_env', False):
                    logger.error("Try to fallback to the base runtime environment.")
                    original_runtime_env = op.runtime_env
                    try:
                        op.runtime_env = None
                        cached_columns = self._run_single_op(op, cached_columns, tracer=tracer)
                    finally:
                        op.runtime_env = original_runtime_env
                else:
                    raise e
        return self

    def _run_single_op(self, op, cached_columns=None, tracer=None):
        # Use cached columns to avoid calling self.data.columns() which breaks pipeline
        if cached_columns is None:
            cached_columns = set(self._cached_columns)

        if op._name in TAGGING_OPS.modules and Fields.meta not in cached_columns:

            def process_batch_arrow(table: pyarrow.Table):
                new_column_data = [{} for _ in range(len(table))]
                new_table = table.append_column(Fields.meta, [new_column_data])
                return new_table

            self.data = self.data.map_batches(
                process_batch_arrow, batch_format="pyarrow", batch_size=DEFAULT_BATCH_SIZE
            )
            cached_columns.add(Fields.meta)

        try:
            batch_size = getattr(op, "batch_size", 1) if op.is_batched_op() else 1
            if isinstance(op, Mapper):
                # Wrap process method with tracer for sample-level collection
                original_process = None
                if tracer and should_trace_op(tracer, op._name):
                    from data_juicer.ops.base_op import wrap_mapper_with_tracer

                    original_process = op.process
                    op.process = wrap_mapper_with_tracer(original_process, op._name, op.text_key, tracer, True)

                try:
                    if op.use_ray_actor():
                        compute = get_compute_strategy(op.__class__, concurrency=op.num_proc)
                        self.data = self.data.map_batches(
                            op.__class__,
                            fn_args=None,
                            fn_kwargs=None,
                            fn_constructor_args=op._init_args,
                            fn_constructor_kwargs=op._init_kwargs,
                            batch_size=batch_size,
                            num_cpus=op.num_cpus,
                            num_gpus=op.num_gpus,
                            compute=compute,
                            batch_format="pyarrow",
                            runtime_env=op.runtime_env,
                        )
                    else:
                        compute = get_compute_strategy(op.process, concurrency=op.num_proc)
                        self.data = self.data.map_batches(
                            op.process,
                            batch_size=batch_size,
                            batch_format="pyarrow",
                            num_cpus=op.num_cpus,
                            num_gpus=op.num_gpus,
                            compute=compute,
                            runtime_env=op.runtime_env,
                        )
                finally:
                    # Restore original process method
                    if tracer and should_trace_op(tracer, op._name) and original_process:
                        op.process = original_process
            elif isinstance(op, Filter):
                # Use cached_columns instead of self.data.columns() to avoid breaking pipeline
                if Fields.stats not in cached_columns:

                    def process_batch_arrow(table: pyarrow.Table):
                        new_column_data = [{} for _ in range(len(table))]
                        new_talbe = table.append_column(Fields.stats, [new_column_data])
                        return new_talbe

                    self.data = self.data.map_batches(
                        process_batch_arrow, batch_format="pyarrow", batch_size=DEFAULT_BATCH_SIZE
                    )
                    cached_columns.add(Fields.stats)
                if op.use_ray_actor():
                    compute = get_compute_strategy(op.__class__, concurrency=op.num_proc)
                    self.data = self.data.map_batches(
                        op.__class__,
                        fn_args=None,
                        fn_kwargs=None,
                        fn_constructor_args=op._init_args,
                        fn_constructor_kwargs=op._init_kwargs,
                        batch_size=batch_size,
                        num_cpus=op.num_cpus,
                        num_gpus=op.num_gpus,
                        compute=compute,
                        batch_format="pyarrow",
                        runtime_env=op.runtime_env,
                    )
                else:
                    compute = get_compute_strategy(op.compute_stats, concurrency=op.num_proc)
                    self.data = self.data.map_batches(
                        op.compute_stats,
                        batch_size=batch_size,
                        batch_format="pyarrow",
                        num_cpus=op.num_cpus,
                        num_gpus=op.num_gpus,
                        compute=compute,
                        runtime_env=op.runtime_env,
                    )
                if op.stats_export_path is not None:
                    self.data.write_json(op.stats_export_path, force_ascii=False)
                # Wrap process method with tracer for sample-level collection
                original_process = None
                if tracer and should_trace_op(tracer, op._name):
                    from data_juicer.ops.base_op import wrap_filter_with_tracer

                    original_process = op.process
                    op.process = wrap_filter_with_tracer(original_process, op._name, tracer, op.is_batched_op())

                try:
                    if op.is_batched_op():
                        # The core computation have been done in compute_stats,
                        # and the filter process only performs simple filtering.
                        # cpu and parallelism are not set here
                        self.data = self.data.map_batches(
                            partial(filter_batch, filter_func=op.process),
                            batch_format="pyarrow",
                            zero_copy_batch=True,
                            batch_size=DEFAULT_BATCH_SIZE,
                            runtime_env=op.runtime_env,
                        )
                    else:
                        self.data = self.data.filter(
                            op.process,
                            runtime_env=op.runtime_env,
                        )
                finally:
                    # Restore original process method
                    if tracer and should_trace_op(tracer, op._name) and original_process:
                        op.process = original_process
            elif isinstance(op, (Deduplicator, Pipeline)):
                self.data = op.run(self.data)
            else:
                logger.error("Ray executor only support Filter, Mapper, Deduplicator and Pipeline OPs for now")
                raise NotImplementedError
        except:  # noqa: E722
            logger.error(f"An error occurred during Op [{op._name}].")
            import traceback

            traceback.print_exc()
            exit(1)

        return cached_columns

    def count(self) -> int:
        return self.data.count()

    @classmethod
    def read(cls, data_format: str, paths: Union[str, List[str]], **kwargs) -> RayDataset:
        if data_format in {"json", "jsonl", "json.gz", "jsonl.gz", "json.zst", "jsonl.zst"}:
            return RayDataset.read_json(paths)
        elif data_format == "webdataset":
            return RayDataset.read_webdataset(paths, **kwargs)
        elif data_format in {
            "parquet",
            "images",
            "parquet_bulk",
            "csv",
            "text",
            "avro",
            "numpy",
            "tfrecords",
            "binary_files",
            "lance",
        }:
            # Optionally drop corrupt/truncated parquet shards before reading.
            # Ray's ParquetDatasource fetches every file's footer up front, so a
            # single bad shard aborts the whole job with ArrowInvalid. Enable by
            # setting DJ_SKIP_CORRUPT_PARQUET=1 (local paths only).
            if data_format in {"parquet", "parquet_bulk"} and os.environ.get(
                "DJ_SKIP_CORRUPT_PARQUET", ""
            ).lower() in ("1", "true", "yes"):
                paths = filter_valid_parquet(paths)
            return getattr(ray.data, f"read_{data_format}")(paths)

    @classmethod
    def read_json(cls, paths: Union[str, List[str]]) -> RayDataset:
        # Note: a temp solution for reading json stream
        # TODO: replace with ray.data.read_json_stream once it is available
        import pyarrow.json as js

        try:
            js.open_json
            return read_json_stream(paths)
        except AttributeError:
            return ray.data.read_json(paths)

    @classmethod
    def read_webdataset(cls, paths: Union[str, List[str]], decoder=None) -> RayDataset:
        if decoder is None:
            decoder = partial(_custom_default_decoder, format="PIL")
        return ray.data.read_webdataset(
            paths, decoder=decoder, file_extensions=["tar"]
        )

    def to_list(self) -> list:
        return self.data.to_pandas().to_dict(orient="records")


class JSONStreamDatasource(ray.data.read_api.ArrowJSONDatasource):
    """
    A temp Datasource for reading json stream.

    Note:

        Depends on a customized `pyarrow` with `open_json` method.
    """

    def _read_stream(self, f: "pyarrow.NativeFile", path: str):
        # Check if open_json is available (PyArrow 20.0.0+)
        try:
            from pyarrow.json import open_json
        except ImportError:
            # Fall back to read_json for older PyArrow versions
            # This will read the entire file into memory, but works with older PyArrow
            import pyarrow.json as js

            try:
                # Read the entire file as a table
                table = js.read_json(f, **self.arrow_json_args)
                if table.num_rows > 0:
                    yield table
            except Exception as e:
                raise ValueError(f"Failed to read JSON file: {path}. Error: {e}") from e
            return

        try:
            reader = open_json(
                f,
                read_options=self.read_options,
                **self.arrow_json_args,
            )
            schema = None
            while True:
                try:
                    batch = reader.read_next_batch()
                    table = pyarrow.Table.from_batches([batch], schema=schema)
                    if schema is None:
                        schema = table.schema
                    yield table
                except StopIteration:
                    return
        except pyarrow.lib.ArrowInvalid as e:
            raise ValueError(f"Failed to read JSON file: {path}.") from e


def read_json_stream(
    paths: Union[str, List[str]],
    *,
    filesystem: Optional["pyarrow.fs.FileSystem"] = None,
    parallelism: int = -1,
    ray_remote_args: Dict[str, Any] = None,
    arrow_open_stream_args: Optional[Dict[str, Any]] = None,
    meta_provider=None,
    partition_filter=None,
    partitioning=ray.data.read_api.Partitioning("hive"),
    include_paths: bool = False,
    ignore_missing_paths: bool = False,
    shuffle: Union[Literal["files"], None] = None,
    file_extensions: Optional[List[str]] = ["json", "jsonl", "json.gz", "jsonl.gz", "json.zst", "jsonl.zst"],
    concurrency: Optional[int] = None,
    override_num_blocks: Optional[int] = None,
    **arrow_json_args,
) -> ray.data.Dataset:
    # Check if open_json is available (PyArrow 20.0.0+)
    # If not, fall back to ray.data.read_json which works with older PyArrow
    try:
        import pyarrow.json as js

        js.open_json  # Check if attribute exists
    except (ImportError, AttributeError):
        # Fall back to standard ray.data.read_json for older PyArrow versions
        # This works with filesystem parameter for S3
        return ray.data.read_json(paths, filesystem=filesystem)

    if meta_provider is None:
        meta_provider = ray.data.read_api.DefaultFileMetadataProvider()

    datasource = JSONStreamDatasource(
        paths,
        arrow_json_args=arrow_json_args,
        filesystem=filesystem,
        open_stream_args=arrow_open_stream_args,
        meta_provider=meta_provider,
        partition_filter=partition_filter,
        partitioning=partitioning,
        ignore_missing_paths=ignore_missing_paths,
        shuffle=shuffle,
        include_paths=include_paths,
        file_extensions=file_extensions,
    )
    return ray.data.read_datasource(
        datasource,
        parallelism=parallelism,
        ray_remote_args=ray_remote_args,
        concurrency=concurrency,
        override_num_blocks=override_num_blocks,
    )
