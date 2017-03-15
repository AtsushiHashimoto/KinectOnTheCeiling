
import cv2, glob, argparse
import numpy as np


def get_args():

    p = argparse.ArgumentParser()
    p.add_argument("-d", "--data_path", type=str, default="../../../Data/")
    p.add_argument("-t", "--target_path", type=str, default="CapturedVideos/")
    args = p.parse_args()

    return args


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


def segment_with_virtual_space(frames, bg_frame):

    vs_rect = np.array([[0, 0], [240, 160]])
    all_vs_px = np.array([[v, h] for v in range(vs_rect[0][0], vs_rect[1][0]) for h in range(vs_rect[0][1], vs_rect[1][1])])
    segmented_frames = []
    for frame in frames:

        processed_frame, _ = segment_human_region(frame, bg_frame)

        if np.any(processed_frame[tuple(all_vs_px.transpose())] > 0):
            segmented_frames.append(processed_frame)

    return segmented_frames


def make_frames(video_fname, out_path):

    cap = cv2.VideoCapture(video_fname)
    ret, frame = cap.read()

    f = 0
    while ret:
        out_fname = out_path + video_fname.split("/")[-1].replace("_segmented.avi", "_%d.png" % f)
        cv2.imwrite(out_fname, frame)
        ret, frame = cap.read()
        f += 1

    cap.release()


def run_and_save_seg_video(video_fname, bg_fname, out_fname):

    cap = cv2.VideoCapture(video_fname)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = frame.shape

    frames = []
    i = 0
    while ret:
        if i % 3 == 0:
            frames.append(frame)
        ret, frame = cap.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        i += 1

    cap.release()

    bg_frame = cv2.imread(bg_fname, cv2.IMREAD_GRAYSCALE)

    seg_frames = segment_with_virtual_space(frames, bg_frame)

    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    seg_out = cv2.VideoWriter(out_fname, fourcc, 30.0, (width, height), isColor=False)
    for seg_frame in seg_frames:
        seg_out.write(seg_frame)

    seg_out.release()

if __name__ == "__main__":

    args = get_args()
    data_path = args.data_path
    target_path = args.target_path
    in_path = data_path + "Main/BodyPartClassification/" + target_path
    out_path = data_path + "Main/BodyPartClassification/CapturedImages/" + "/".join(target_path.split("/")[1:])
    bg_fname = in_path + "bg.png"
    video_fnames = glob.glob(in_path + "*_raw.avi")

    for video_fname in video_fnames:
        print("/".join(video_fname.split("/")[-2:]))
        seg_video_fname = video_fname.replace("raw.avi", "segmented.avi")
        run_and_save_seg_video(video_fname, bg_fname, seg_video_fname)
        make_frames(seg_video_fname, out_path)

