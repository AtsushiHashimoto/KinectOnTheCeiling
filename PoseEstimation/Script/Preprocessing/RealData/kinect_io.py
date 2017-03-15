# coding: utf-8

# An example using startStreams

import numpy as np
import cv2
import sys, time
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame

try:
    from pylibfreenect2 import OpenCLPacketPipeline
    pipeline = OpenCLPacketPipeline()
except:
    from pylibfreenect2 import CpuPacketPipeline
    pipeline = CpuPacketPipeline()

enable_depth = True

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

types = 0
if enable_depth:
    types |= (FrameType.Ir | FrameType.Depth)
listener = SyncMultiFrameListener(types)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

start = time.time()

time.sleep(10)
frames = listener.waitForNewFrame()

depth = frames["depth"]
d_array = depth.asarray() / 4500.
cv2.imshow("depth", d_array)

listener.release(frames)

key = cv2.waitKey(delay=1)
cv2.imwrite("~/Desktop/test.png", (d_array * 255).astype(np.uint8))

device.stop()
device.close()

sys.exit(0)
