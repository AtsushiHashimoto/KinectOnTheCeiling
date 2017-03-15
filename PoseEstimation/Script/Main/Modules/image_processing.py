
import cv2
import numpy as np

__all__ = ["extract_human_region", "human_rectangle", "segment_human_region"]


def extract_human_region(label_px):

    sum_of_px_vals = np.sum(label_px, axis=2)
    foreground_pixels = np.array(np.where((sum_of_px_vals > 32) & (sum_of_px_vals < 255*3 - 32)), dtype=np.uint16).transpose()

    return foreground_pixels


def human_rectangle(label_px):

    foreground_pixels = extract_human_region(label_px)
    min_v = foreground_pixels[:, 0].min()
    max_v = foreground_pixels[:, 0].max()
    min_u = foreground_pixels[:, 1].min()
    max_u = foreground_pixels[:, 1].max()

    return min_v, max_v, min_u, max_u


def segment_human_region(raw_frame, bg_frame):

    raw_array = raw_frame * 1000. / 255.
    bg_frame = bg_frame * 1000. / 255.
    height, width = raw_array.shape
    sub_array = np.abs(raw_array - bg_frame)
    fg_mask = np.zeros((height, width), dtype=np.uint8)
    fg_mask[np.where(sub_array > 10)] = 1
    fg_mask[np.where(raw_array == 0)] = 0

    neighbor8 = np.ones((3, 3), dtype=np.uint8)
    eroded_array = cv2.erode(fg_mask, neighbor8, iterations=4)
    processed_fg_mask = cv2.dilate(eroded_array, neighbor8, iterations=4)

    output = cv2.connectedComponentsWithStats(processed_fg_mask.astype(np.uint8), connectivity=8)
    human_mask = np.zeros(raw_array.shape, dtype=np.uint8)
    n_most_popular_label_px = 0
    human_label = 1
    for label in np.unique(output[1])[1:]:
        n_label_px = np.where(output[1]==label)[0].shape[0]
        if n_label_px > n_most_popular_label_px:
            human_label = label
            n_most_popular_label_px = n_label_px
    human_mask[np.where(output[1] == human_label)] = 1

    processed_array = (human_mask * raw_array * 255. / 1000. ).astype(np.uint8)
    human_mask = np.repeat(human_mask * 127, 3).reshape((height, width, 3)).astype(np.uint8)

    return processed_array, human_mask

