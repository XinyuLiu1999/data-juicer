import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ..base_op import OPERATORS, Filter

SIDE_NAMES = ('top', 'bottom', 'left', 'right')


def _get_strip(img, side, thickness_h, thickness_w):
    """Extract a border strip from the given side of the image."""
    if side == 'top':
        return img[:thickness_h, :, :]
    elif side == 'bottom':
        return img[-thickness_h:, :, :]
    elif side == 'left':
        return img[:, :thickness_w, :]
    else:
        return img[:, -thickness_w:, :]


def _get_adjacent_strip(img, side, thin_px, gap_factor=2, width_factor=4):
    """Extract a strip adjacent (inward) to the border for comparison.

    The adjacent strip starts at ``thin_px * gap_factor`` from the edge and
    extends to ``thin_px * width_factor``.
    """
    h, w = img.shape[:2]
    start = thin_px * gap_factor
    end = thin_px * width_factor
    if side == 'bottom':
        return img[-end:-start, :, :]
    elif side == 'top':
        return img[start:end, :, :]
    elif side == 'left':
        return img[:, start:end, :]
    else:
        return img[:, -end:-start, :]


def _has_sharp_edge(img, side, bh, bw, edge_threshold=30):
    """Return True if there is a sharp row/col brightness transition near
    the border, indicating an artificial bar boundary."""
    h, w = img.shape[:2]
    if side in ('top', 'bottom'):
        depth = min(bh * 3, h // 2)
    else:
        depth = min(bw * 3, w // 2)

    if side == 'top':
        vals = [np.mean(img[y, :, :]) for y in range(depth)]
    elif side == 'bottom':
        vals = [np.mean(img[h - 1 - y, :, :]) for y in range(depth)]
    elif side == 'left':
        vals = [np.mean(img[:, x, :]) for x in range(depth)]
    else:
        vals = [np.mean(img[:, w - 1 - x, :]) for x in range(depth)]

    if len(vals) < 2:
        return False
    max_jump = max(abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1))
    return max_jump > edge_threshold


def _uniformity_contrast(img, side, contrast_threshold=3.0,
                        max_bar_variance=100.0):
    """Return True if the thin outermost border is much more uniform (low
    variance) than the adjacent image content — a strong signal that the
    border is an artificial flat-colour strip rather than scene content.

    An additional ``max_bar_variance`` guard ensures the thin strip itself
    is actually flat.  Natural dark regions (night sky, dark surfaces) often
    have variance well above this floor even though they may be *relatively*
    more uniform than the busier content beside them."""
    h, w = img.shape[:2]
    thin_px = max(int((h if side in ('top', 'bottom') else w) * 0.01), 3)
    thin = _get_strip(img, side, thin_px, thin_px)
    adj = _get_adjacent_strip(img, side, thin_px)

    thin_var = np.mean(np.var(thin.reshape(-1, 3), axis=0))
    adj_var = np.mean(np.var(adj.reshape(-1, 3), axis=0))

    # The strip must be both (a) flat in absolute terms AND (b) much
    # more uniform than the adjacent content.
    if thin_var > max_bar_variance:
        return False
    return adj_var / (thin_var + 0.01) > contrast_threshold


def _adjacent_brightness(img, side):
    """Return the mean brightness of image content adjacent to the border,
    skipping the border strip itself."""
    h, w = img.shape[:2]
    depth_h = max(int(h * 0.05), 5)
    depth_w = max(int(w * 0.05), 5)
    if side == 'bottom':
        adj = img[-(depth_h * 3):-(depth_h), :, :]
    elif side == 'top':
        adj = img[depth_h:depth_h * 3, :, :]
    elif side == 'left':
        adj = img[:, depth_w:depth_w * 3, :]
    else:
        adj = img[:, -(depth_w * 3):-(depth_w), :]
    return np.mean(adj)


def calc_border_score(image,
                      border_ratio=0.05,
                      variance_threshold=50,
                      uniformity_threshold=0.85,
                      light_bg_threshold=200,
                      neutral_max_spread=15,
                      dark_threshold=80,
                      light_threshold=220,
                      bar_ratio_threshold=0.90,
                      edge_brightness_threshold=30,
                      uniformity_contrast_threshold=3.0,
                      light_bar_adjacent_max_brightness=180,
                      center_variance_threshold=500):
    """Detect artificial borders / bars while preserving product-style images.

    The function performs three layers of detection:

    1. **Per-side uniform-frame detection** – each border (top / bottom /
       left / right) is checked independently for low variance + high
       colour uniformity.  A non-light-neutral uniform frame on ≥ 3 sides
       is flagged.
    2. **Dark / light bar detection** – each side is checked for a
       concentration of very dark (< *dark_threshold*) or very bright
       (> *light_threshold*) pixels, with a **multi-scale scan** to catch
       thin bars missed at the default *border_ratio*.
    3. **Confirmation** – candidate bars are validated via *sharp edge*
       detection (abrupt brightness transition) **or** *uniformity contrast*
       (bar variance ≪ adjacent content variance).  Light bars additionally
       require the adjacent content to be darker than
       *light_bar_adjacent_max_brightness* to avoid flagging product-style
       white backgrounds.

    Product-style images (all uniform sides are light-neutral + centre has
    rich content) are explicitly **preserved**.

    Args:
        image: PIL Image object.
        border_ratio: Fraction of width / height used as border strip.
        variance_threshold: Max mean per-channel variance for "uniform".
        uniformity_threshold: Min fraction of border pixels near the mean.
        light_bg_threshold: Min brightness for "light background".
        neutral_max_spread: Max R/G/B channel spread for "neutral".
        dark_threshold: Max mean brightness for a "dark" pixel.
        light_threshold: Min mean brightness for a "light" pixel.
        bar_ratio_threshold: Min dark/light pixel fraction for a bar.
        edge_brightness_threshold: Min row-to-row brightness jump for a
            sharp edge.
        uniformity_contrast_threshold: Min ratio (adjacent var / bar var)
            to confirm bar via uniformity contrast.
        light_bar_adjacent_max_brightness: Max adjacent brightness for a
            light bar to be confirmed (prevents product-bg false positives).
        center_variance_threshold: Min centre variance for "has content".

    Returns:
        int: 1 if an artificial frame or bar is detected (should filter),
             0 if the image is clean or product-style (should keep).
    """
    img = np.array(image.convert('RGB'), dtype=np.float32)
    h, w, _ = img.shape

    bh = max(int(h * border_ratio), 2)
    bw = max(int(w * border_ratio), 2)

    # Abort for very small images
    if h < 8 or w < 8:
        return 0

    # ------------------------------------------------------------------
    # 1. Per-side analysis at default border_ratio
    # ------------------------------------------------------------------
    borders = {
        'top':    img[:bh, :, :],
        'bottom': img[-bh:, :, :],
        'left':   img[:, :bw, :],
        'right':  img[:, -bw:, :],
    }

    side_info = {}      # per-side metadata
    frame_sides = []    # sides that are uniform
    bar_sides = []      # sides that look like dark/light bars

    for name, strip in borders.items():
        pixels = strip.reshape(-1, 3)
        mean_color = np.mean(pixels, axis=0)
        var = np.mean(np.var(pixels, axis=0))

        # Uniformity check
        distances = np.sqrt(np.sum((pixels - mean_color) ** 2, axis=1))
        uniform_ratio = float(np.mean(distances < 30))
        is_uniform = (var < variance_threshold
                      and uniform_ratio > uniformity_threshold)

        # Dark / light bar check (brightness-based)
        pixel_brightness = np.mean(pixels, axis=1)
        dark_ratio = float(np.mean(pixel_brightness < dark_threshold))
        light_ratio = float(np.mean(pixel_brightness > light_threshold))
        is_dark_bar = dark_ratio > bar_ratio_threshold
        is_light_bar = light_ratio > bar_ratio_threshold

        # Light-neutral background check (product-style guard)
        brightness = float(np.mean(mean_color))
        spread = float(np.max(mean_color) - np.min(mean_color))
        is_light_neutral = (brightness > light_bg_threshold
                            and spread < neutral_max_spread)

        side_info[name] = {
            'is_uniform': is_uniform,
            'is_dark_bar': is_dark_bar,
            'is_light_bar': is_light_bar,
            'is_light_neutral': is_light_neutral,
        }

        if is_uniform:
            frame_sides.append(name)
        if is_dark_bar or is_light_bar:
            bar_sides.append(name)

    # ------------------------------------------------------------------
    # 2. Multi-scale thin-bar scan
    # ------------------------------------------------------------------
    thin_bar_min_px = 3
    thin_bar_ratios = [0.01, 0.02, 0.03, 0.04, 0.05]

    for side in SIDE_NAMES:
        if side in bar_sides:
            continue
        for ratio in thin_bar_ratios:
            th = max(int(h * ratio), thin_bar_min_px)
            tw = max(int(w * ratio), thin_bar_min_px)
            strip = _get_strip(img, side, th, tw)
            pixels = strip.reshape(-1, 3)
            brightness = np.mean(pixels, axis=1)

            dr = float(np.mean(brightness < dark_threshold))
            if dr > bar_ratio_threshold:
                bar_sides.append(side)
                side_info[side]['is_dark_bar'] = True
                break

            lr = float(np.mean(brightness > light_threshold))
            if lr > bar_ratio_threshold:
                bar_sides.append(side)
                side_info[side]['is_light_bar'] = True
                break

    # ------------------------------------------------------------------
    # 3. Confirm candidate bars (sharp edge OR uniformity contrast)
    # ------------------------------------------------------------------
    # Both confirmation paths now also check absolute brightness of the
    # thin strip to ensure it is in true bar territory, not just a
    # natural dark/light region of the scene.
    #
    # Sharp-edge confirmation: the strip must have a clear brightness
    #   boundary AND its mean brightness must be below dark_threshold/2
    #   (for dark bars) or above light_threshold (for light bars).
    #   Natural scene transitions (dark wall → bright window) have
    #   moderate brightness (50–80) that real bars don't.
    #
    # Uniformity-contrast confirmation: the strip must be flat (low var),
    #   have high contrast ratio vs adjacent content, AND have very low
    #   brightness.
    confirmed_bars = []
    for s in bar_sides:
        has_edge = _has_sharp_edge(img, s, bh, bw, edge_brightness_threshold)
        has_contrast = _uniformity_contrast(
            img, s, uniformity_contrast_threshold)

        thin_px = max(int(
            (h if s in ('top', 'bottom') else w) * 0.01), 3)
        thin_strip = _get_strip(img, s, thin_px, thin_px)
        strip_brightness = float(np.mean(thin_strip))

        is_bar_dark = strip_brightness < (dark_threshold * 0.7)
        is_bar_light = strip_brightness > light_threshold

        if has_edge and (is_bar_dark or is_bar_light):
            confirmed = True
        elif has_contrast and (is_bar_dark or is_bar_light):
            confirmed = True
        else:
            confirmed = False

        if not confirmed:
            continue

        # Extra guard for light bars: adjacent content must be darker
        if side_info[s].get('is_light_bar', False):
            adj_bright = _adjacent_brightness(img, s)
            if adj_bright > light_bar_adjacent_max_brightness:
                continue
        confirmed_bars.append(s)

    bar_sides = confirmed_bars

    # ------------------------------------------------------------------
    # 4. Centre content check (for product-style detection)
    # ------------------------------------------------------------------
    center_crop = img[h // 4:3 * h // 4, w // 4:3 * w // 4, :]
    center_var = float(np.mean(np.var(center_crop.reshape(-1, 3), axis=0)))
    has_center_content = center_var > center_variance_threshold

    # ------------------------------------------------------------------
    # 5. Decision logic
    # ------------------------------------------------------------------
    # Full uniform-colour frame (≥ 3 non-light-neutral sides)
    uniform_non_light = [
        s for s in frame_sides if not side_info[s]['is_light_neutral']
    ]
    has_full_frame = len(uniform_non_light) >= 3

    # Artificial bar detected on at least one side
    has_bar = len(bar_sides) > 0

    # Product-style image: all uniform sides are light-neutral + centre has
    # real content  →  preserve
    all_uniform_light = (
        all(side_info[s]['is_light_neutral'] for s in frame_sides)
        if frame_sides else False
    )
    is_product_style = (
        all_uniform_light and has_center_content and len(frame_sides) >= 2
    )

    should_filter = (has_full_frame or has_bar) and not is_product_style
    return 1 if should_filter else 0


@OPERATORS.register_module('image_border_variance_filter')
class ImageBorderVarianceFilter(Filter):
    """Filter to detect and remove images with artificial borders or bars
    while preserving product-style images with white/light backgrounds.

    Registered as ``image_border_variance_filter``.

    Unlike simple variance-based border checks, this filter:

    - Checks each edge **independently** so partial bars (e.g. a single
      watermark strip at the bottom) are caught.
    - Uses **multi-scale scanning** to detect thin bars that are narrower
      than the default border strip.
    - **Confirms** candidate bars via sharp-edge detection or
      uniformity-contrast analysis to avoid false positives on natural
      dark/light scene content (e.g. night sky, overcast sky).
    - Handles both **dark bars** (black letterboxing, dark watermark
      strips) and **light bars** (white info strips, bright frames).
    - Explicitly **preserves** product-style images that have uniform
      white / off-white / light-grey backgrounds with content in the
      centre (typical of e-commerce product photos, icons, logos).

    Use cases:
        - Filter stock-photo watermark bars (Alamy, Shutterstock, etc.)
        - Filter letterboxed / pillarboxed video frames
        - Filter images with decorative coloured frames
        - Preserve product photos, icons, and logos on light backgrounds
    """

    _default_kwargs = {
        'border_ratio': 0.05,
        'variance_threshold': 50,
        'uniformity_threshold': 0.85,
        'light_bg_threshold': 200,
        'neutral_max_spread': 15,
        'dark_threshold': 80,
        'light_threshold': 220,
        'bar_ratio_threshold': 0.90,
        'edge_brightness_threshold': 30,
        'uniformity_contrast_threshold': 3.0,
        'light_bar_adjacent_max_brightness': 180,
        'center_variance_threshold': 500,
    }

    def __init__(self,
                 border_ratio=0.05,
                 variance_threshold=50,
                 uniformity_threshold=0.85,
                 light_bg_threshold=200,
                 neutral_max_spread=15,
                 dark_threshold=80,
                 light_threshold=220,
                 bar_ratio_threshold=0.90,
                 edge_brightness_threshold=30,
                 uniformity_contrast_threshold=3.0,
                 light_bar_adjacent_max_brightness=180,
                 center_variance_threshold=500,
                 any_or_all='any',
                 *args,
                 **kwargs):
        """Initialise the ImageBorderVarianceFilter.

        Args:
            border_ratio (float): Fraction of width/height used as the
                default border strip.  Default ``0.05`` (5 %).
            variance_threshold (float): Max mean per-channel variance for
                a border to be considered "uniform".  Lower → stricter.
                Default ``50``.
            uniformity_threshold (float): Min fraction of border pixels
                within 30 units of the mean colour.  Default ``0.85``.
            light_bg_threshold (float): Min mean brightness (0–255) for a
                border to qualify as a "light background".  Default ``200``.
            neutral_max_spread (float): Max R/G/B spread to count as a
                neutral (grey / white) colour.  Default ``15``.
            dark_threshold (float): Max mean brightness for a pixel to be
                counted as "dark".  Default ``80``.
            light_threshold (float): Min mean brightness for a pixel to be
                counted as "light".  Default ``220``.
            bar_ratio_threshold (float): Min fraction of dark/light pixels
                in a strip to flag it as a bar.  Default ``0.90``.
            edge_brightness_threshold (float): Min row-to-row brightness
                jump to confirm a sharp edge.  Default ``30``.
            uniformity_contrast_threshold (float): Min ratio of adjacent
                variance to bar variance to confirm via uniformity
                contrast.  Default ``3.0``.
            light_bar_adjacent_max_brightness (float): Max adjacent content
                brightness for a light bar to be confirmed (avoids
                product-bg false positives).  Default ``180``.
            center_variance_threshold (float): Min centre-crop variance to
                consider the image as having real content.  Default ``500``.
            any_or_all (str): Strategy for multi-image samples.
                ``'any'``: keep if *any* image passes.
                ``'all'``: keep only if *all* images pass.
        """
        super().__init__(*args, **kwargs)
        if any_or_all not in ('any', 'all'):
            raise ValueError("any_or_all must be 'any' or 'all'")
        self.any = (any_or_all == 'any')

        self.border_ratio = border_ratio
        self.variance_threshold = variance_threshold
        self.uniformity_threshold = uniformity_threshold
        self.light_bg_threshold = light_bg_threshold
        self.neutral_max_spread = neutral_max_spread
        self.dark_threshold = dark_threshold
        self.light_threshold = light_threshold
        self.bar_ratio_threshold = bar_ratio_threshold
        self.edge_brightness_threshold = edge_brightness_threshold
        self.uniformity_contrast_threshold = uniformity_contrast_threshold
        self.light_bar_adjacent_max_brightness = \
            light_bar_adjacent_max_brightness
        self.center_variance_threshold = center_variance_threshold

    def compute_stats_single(self, sample, context=False):
        # Skip if already computed
        if StatsKeys.image_border_variance_scores in sample[Fields.stats]:
            return sample

        # Handle missing or empty image key
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][
                StatsKeys.image_border_variance_scores] = []
            return sample

        # Load images
        keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, keys, load_image,
            mm_bytes_key=self.image_bytes_key)

        # Score each image: 1 = has artificial border, 0 = clean / product
        sample[Fields.stats][
            StatsKeys.image_border_variance_scores] = [
            calc_border_score(
                images[k],
                border_ratio=self.border_ratio,
                variance_threshold=self.variance_threshold,
                uniformity_threshold=self.uniformity_threshold,
                light_bg_threshold=self.light_bg_threshold,
                neutral_max_spread=self.neutral_max_spread,
                dark_threshold=self.dark_threshold,
                light_threshold=self.light_threshold,
                bar_ratio_threshold=self.bar_ratio_threshold,
                edge_brightness_threshold=self.edge_brightness_threshold,
                uniformity_contrast_threshold=self
                    .uniformity_contrast_threshold,
                light_bar_adjacent_max_brightness=self
                    .light_bar_adjacent_max_brightness,
                center_variance_threshold=self.center_variance_threshold,
            )
            for k in keys
        ]
        return sample

    def process_single(self, sample):
        scores = sample[Fields.stats][
            StatsKeys.image_border_variance_scores]
        if not scores:
            return True

        # 0 = clean (keep), 1 = has border (filter out)
        keep = np.array([v == 0 for v in scores])
        return keep.any() if self.any else keep.all()