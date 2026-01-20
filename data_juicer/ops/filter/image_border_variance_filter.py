import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ..base_op import OPERATORS, Filter


def calc_border_variance(image, border_width=10, variance_threshold=100,
                        white_threshold=200, black_threshold=55):
    """
    Detect if image has white or black borders by checking edge uniformity
    and color.
    
    Args:
        image: PIL Image object
        border_width: Width of the border region to analyze (in pixels)
        variance_threshold: Threshold for determining uniform edges.
                          Lower values = stricter detection.
        white_threshold: Minimum mean pixel value to consider as white.
                        Default 200 (out of 255).
        black_threshold: Maximum mean pixel value to consider as black.
                        Default 55 (out of 255).
    
    Returns:
        int: 1 if white/black border detected (should be filtered out), 
             0 if no white/black border (should be kept)
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = img_array[:, :, np.newaxis]
    
    height, width = img_array.shape[:2]
    
    # Adjust border_width if image is too small
    border_width = min(border_width, height // 4, width // 4)
    if border_width < 1:
        return 0  # Image too small, assume no border
    
    # Extract border regions
    edges = [
        img_array[:border_width, :, :],  # top
        img_array[-border_width:, :, :],  # bottom
        img_array[:, :border_width, :],  # left
        img_array[:, -border_width:, :]  # right
    ]
    
    # Check if any edge is uniform AND white or black
    for edge in edges:
        pixels = edge.reshape(-1, img_array.shape[-1])
        
        # Calculate variance per channel and sum them
        num_channels = pixels.shape[1]
        variance = sum(np.var(pixels[:, c]) for c in range(num_channels))
        
        # Check if edge is uniform (low variance)
        if variance < variance_threshold:
            # Calculate mean brightness across all channels
            mean_value = pixels.mean()
            
            # Check if it's white or black
            is_white = mean_value >= white_threshold
            is_black = mean_value <= black_threshold
            
            if is_white or is_black:
                return 1  # White or black border detected
    
    return 0  # No white/black border detected


@OPERATORS.register_module('image_border_variance_filter')
class ImageBorderVarianceFilter(Filter):
    """Filter to detect and remove images with white or black uniform borders.

    This filter analyzes each edge of the image to identify uniform white or
    black borders. Images are flagged if ANY edge shows uniformity (low variance)
    AND is white or black in color.

    Use cases:
    - Filter out images with white borders/frames
    - Filter out images with black borders/letterboxing
    - Remove scanned documents with white margins
    - Remove screenshots with black bars
    """

    def __init__(self,
                 border_width=10,
                 variance_threshold=100,
                 white_threshold=200,
                 black_threshold=55,
                 any_or_all='any',
                 *args,
                 **kwargs):
        """
        Initialize the ImageBorderVarianceFilter.

        Args:
            border_width: Width of the border region to analyze in pixels.
                Default is 10 pixels.
            variance_threshold: Threshold for determining uniform edges.
                Lower values = stricter detection (more images filtered).
                Higher values = more lenient (fewer images filtered).
                Default is 100. Typical range: 50-200.
            white_threshold: Minimum mean pixel value (0-255) to consider
                as white. Default is 200. Higher = stricter white detection.
            black_threshold: Maximum mean pixel value (0-255) to consider
                as black. Default is 55. Lower = stricter black detection.
            any_or_all: Strategy for handling multiple images in a sample.
                'any': Keep sample if any image has no white/black border.
                'all': Keep sample only if all images have no white/black border.
        """
        super().__init__(*args, **kwargs)
        if any_or_all not in ['any', 'all']:
            raise ValueError("any_or_all must be 'any' or 'all'")
        self.any = any_or_all == 'any'
        self.border_width = border_width
        self.variance_threshold = variance_threshold
        self.white_threshold = white_threshold
        self.black_threshold = black_threshold

    def compute_stats_single(self, sample, context=False):
        # Check if stats already computed
        if StatsKeys.image_border_variance_scores in sample[Fields.stats]:
            return sample

        # Handle missing or empty image key
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_border_variance_scores] = []
            return sample

        # Load images
        keys = sample[self.image_key]
        sample, images = load_data_with_context(sample,
                                                context,
                                                keys,
                                                load_image,
                                                mm_bytes_key=self.image_bytes_key)

        # Calculate border detection for each image
        # Returns 1 (has white/black border) or 0 (no white/black border)
        sample[Fields.stats][StatsKeys.image_border_variance_scores] = [
            calc_border_variance(images[k], 
                               self.border_width, 
                               self.variance_threshold,
                               self.white_threshold,
                               self.black_threshold) 
            for k in keys
        ]
        return sample

    def process_single(self, sample):
        vals = sample[Fields.stats][StatsKeys.image_border_variance_scores]
        if not vals:
            return True

        # vals contains 1 (has white/black border - should filter) 
        # or 0 (no white/black border - should keep)
        keep = [v == 0 for v in vals]
        
        keep = np.array(keep)
        # any: keep if ANY image has no white/black border
        # all: keep only if ALL images have no white/black border
        return keep.any() if self.any else keep.all()