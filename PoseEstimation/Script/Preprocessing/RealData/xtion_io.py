# -*- coding=utf8 -*-

import cv2, time, argparse
import numpy as np
from primesense import openni2


class XtionIO:
    # 入力の際の深度の単位はmm

    def __init__(self, data_path="../../../Data/Main/BodyPartClassification/CapturedVideos/"):
        self.videos_path = data_path
        self.depth_stream = None
        self.frame_shape = None
        self.fourcc = None

    def initialize(self):
        openni2.initialize()     # can also accept the path of the OpenNI redistribution
        dev = openni2.Device.open_any()
        depth_stream = dev.create_depth_stream()
        depth_stream.start()
        rgb_stream = dev.create_color_stream()
        rgb_stream.start()
        frame = depth_stream.read_frame()
        self.depth_stream = depth_stream
        self.rgb_stream = rgb_stream
        self.frame_shape = (frame.width, frame.height)
        self.fourcc = cv2.VideoWriter_fourcc(*'FFV1')

    def capture_bg_rgb_image(self, bg_img_name):

        bg_img_filename = self.videos_path + bg_img_name

        frame = self.rgb_stream.read_frame()
        frame_data = frame.get_buffer_as_uint8()

        bg_array = np.ndarray((self.frame_shape[1], self.frame_shape[0], 3),
                              dtype=np.uint8, buffer=frame_data)[:, :, ::-1]

        cv2.imwrite(bg_img_filename, bg_array.astype(np.uint8))

    def capture_bg_depth_image(self, bg_img_name):

        time.sleep(5)
        bg_img_filename = self.videos_path + bg_img_name

        frame = self.depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()

        bg_array = np.ndarray(self.frame_shape[::-1], dtype=np.uint16, buffer=frame_data)

        cv2.imwrite(bg_img_filename, (bg_array / 10000. * 255).astype(np.uint8))

    def capture_rgb_video(self, video_name):
        time.sleep(10)
        video_out_filename = self.videos_path + video_name
        raw_out = cv2.VideoWriter(video_out_filename, self.fourcc, 30.0, self.frame_shape, isColor=True)
        start = time.time()
        while True:

            frame = self.rgb_stream.read_frame()
            frame_data = frame.get_buffer_as_uint8()

            raw_array = np.ndarray((self.frame_shape[1], self.frame_shape[0], 3), dtype=np.uint8, buffer=frame_data)[:, :, ::-1]
            cv2.imshow("Raw", raw_array.astype(np.uint8))
            raw_out.write(raw_array.astype(np.uint8))

            if (cv2.waitKey(1) & 0xFF == ord('q')) or (time.time() - start > 10):
                break

        raw_out.release()
        cv2.destroyAllWindows()

    def capture_depth_video(self, video_name):

        video_out_filename = self.videos_path + video_name
        raw_out = cv2.VideoWriter(video_out_filename, self.fourcc, 30.0, self.frame_shape, isColor=False)
        while True:

            frame = self.depth_stream.read_frame()
            frame_data = frame.get_buffer_as_uint16()

            raw_array = np.ndarray(self.frame_shape[::-1], dtype=np.uint16, buffer=frame_data)
            rounded_array = (raw_array / 10000. * 255).astype(np.uint8)
            cv2.imshow("Raw", rounded_array)

            raw_out.write(rounded_array)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        raw_out.release()
        cv2.destroyAllWindows()

    def release(self):
        self.depth_stream.stop()
        self.rgb_stream.stop()
        openni2.unload()


