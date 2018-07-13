import argparse
import asyncio
import json
import logging
import math
import os
import time
import wave

import cv2
import numpy
from aiohttp import web

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import (AudioFrame, AudioStreamTrack, VideoFrame,
                                 VideoStreamTrack)


# --- for yolo v3 ---
from ctypes import *
#import math
import random

# --- multi process --
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Value, Array, RawArray

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)



free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


# --- add for save image, frame and label --
save_image = lib.save_image
save_image.argtypes = [IMAGE, c_char_p]
draw_box_width = lib.draw_box_width
draw_box_width.argtypes = [IMAGE, c_int, c_int, c_int, c_int, c_int, c_float, c_float, c_float]

load_alphabet = lib.load_alphabet
load_alphabet.restype = POINTER(POINTER(IMAGE))

get_label = lib.get_label
get_label.restype = IMAGE
get_label.argtypes = [POINTER(POINTER(IMAGE)), c_char_p, c_int]
draw_label = lib.draw_label
draw_label.argtypes = [IMAGE, c_int, c_int, IMAGE, POINTER(c_float)]

# --- convert arra <--> darknet IMAGE ---
def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = c_array(c_float, arr)
    im = IMAGE(w,h,c,data)
    return im

def image_to_array(im):
    size = im.w * im.h * im.c
    data_pointer = cast(im.data, POINTER(c_float))
    arr = numpy.ctypeslib.as_array(data_pointer,shape=(size,))
    arrU8 = (arr * 255.0).astype(numpy.uint8)
    arrU8 = arrU8.reshape(im.c, im.h, im.w)
    arrU8 = arrU8.transpose(1,2,0)
    return arrU8

def arrayConvTest(arr):
    w = arr.shape[0]
    h = arr.shape[1]
    c = arr.shape[2]

    # --- try 4 ---
    # https://stackoverflow.com/questions/23930671/how-to-create-n-dim-numpy-array-from-a-pointer
    arr = (arr/255.0).flatten()
    data = c_array(c_float, arr)
    size = w * h * c
    data_pointer = cast(data, POINTER(c_float))
    arr2 = numpy.ctypeslib.as_array(data_pointer,shape=(size,))
    arr3 = (arr2 * 255.0).astype(numpy.uint8)
    newArr = arr3.reshape(w, h, c)

    return newArr

# --- for yolo v3 ---

# --- for aiortc ---
ROOT = os.path.dirname(__file__)
AUDIO_OUTPUT_PATH = os.path.join(ROOT, 'output.wav')
AUDIO_PTIME = 0.020  # 20ms audio packetization


def frame_from_bgr(data_bgr):
    data_yuv = cv2.cvtColor(data_bgr, cv2.COLOR_BGR2YUV_YV12)
    return VideoFrame(width=data_bgr.shape[1], height=data_bgr.shape[0], data=data_yuv.tobytes())


def frame_from_gray(data_gray):
    data_bgr = cv2.cvtColor(data_gray, cv2.COLOR_GRAY2BGR)
    data_yuv = cv2.cvtColor(data_bgr, cv2.COLOR_BGR2YUV_YV12)
    return VideoFrame(width=data_bgr.shape[1], height=data_bgr.shape[0], data=data_yuv.tobytes())


def frame_to_bgr(frame):
    data_flat = numpy.frombuffer(frame.data, numpy.uint8)
    data_yuv = data_flat.reshape((math.ceil(frame.height * 12 / 8), frame.width))
    return cv2.cvtColor(data_yuv, cv2.COLOR_YUV2BGR_YV12)


class AudioFileTrack(AudioStreamTrack):
    def __init__(self, path):
        self.last = None
        self.reader = wave.open(path, 'rb')
        self.frames_per_packet = int(self.reader.getframerate() * AUDIO_PTIME)

    async def recv(self):
        # as we are reading audio from a file and not using a "live" source,
        # we need to control the rate at which audio is sent
        if self.last:
            now = time.time()
            await asyncio.sleep(self.last + AUDIO_PTIME - now)
        self.last = time.time()

        return AudioFrame(
            channels=self.reader.getnchannels(),
            data=self.reader.readframes(self.frames_per_packet),
            sample_rate=self.reader.getframerate())


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, transform):
        self.counter = 0
        self.received = asyncio.Queue(maxsize=1)
        self.transform = transform
        self.tryDetectCounter = 0

    async def recv(self):
        frame = await self.received.get()

        self.counter += 1

        # --- try reduce frame -- NG
        #if (self.counter % 2) == 1:
        #    return None

        if self.transform == 'yolov3_worker':
            with workerBusy.get_lock():
                if (workerBusy.value == 0):
                    # --- kick detect --
                    workerBusy.value = 1
                    img = frame_to_bgr(frame)
                    future = executor.submit(workerDetect, img)
                    future.add_done_callback(onDetected)
                    self.tryDetectCounter += 1
            
            # --- draw last detect ---
            img = frame_to_bgr(frame)
            #rows, cols, _ = img.shape            
            for result in lastResults:
                if result.bottom == 0:
                    break
                name = str(meta.names[result.cid])
                img = cv2.rectangle(img, (result.left, result.top), (result.right, result.bottom), (255, 128,0), 3, 4)
                img = cv2.putText(img, name, (result.left, result.top), cv2.FONT_HERSHEY_PLAIN, 2, (255, 128, 0), 2, 4)

            # --- count ---
            dcount = str(self.tryDetectCounter)
            img = cv2.putText(img, dcount, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 128, 128), 2, 4)

            newFrame = frame_from_bgr(img)
            return newFrame
                       
        else:
            # return raw frame
            return frame


async def consume_audio(track):
    """
    Drain incoming audio and write it to a file.
    """
    writer = None

    try:
        while True:
            frame = await track.recv()
            if writer is None:
                writer = wave.open(AUDIO_OUTPUT_PATH, 'wb')
                writer.setnchannels(frame.channels)
                writer.setframerate(frame.sample_rate)
                writer.setsampwidth(frame.sample_width)
            writer.writeframes(frame.data)
    finally:
        if writer is not None:
            writer.close()


async def consume_video(track, local_video):
    """
    Drain incoming video, and echo it back.
    """
    while True:
        frame = await track.recv()

        # we are only interested in the latest frame
        if local_video.received.full():
            await local_video.received.get()

        await local_video.received.put(frame)


async def index(request):
    content = open(os.path.join(ROOT, 'index.html'), 'r').read()
    return web.Response(content_type='text/html', text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, 'client.js'), 'r').read()
    return web.Response(content_type='application/javascript', text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(
        sdp=params['sdp'],
        type=params['type'])

    pc = RTCPeerConnection()
    pc._consumers = []
    pcs.append(pc)

    # prepare local media
    local_audio = AudioFileTrack(path=os.path.join(ROOT, 'demo-instruct.wav'))
    local_video = VideoTransformTrack(transform=params['video_transform'])

    @pc.on('datachannel')
    def on_datachannel(channel):
        @channel.on('message')
        def on_message(message):
            channel.send('pong')

    @pc.on('track')
    def on_track(track):
        if track.kind == 'audio':
            pc.addTrack(local_audio)
            pc._consumers.append(asyncio.ensure_future(consume_audio(track)))
        elif track.kind == 'video':
            pc.addTrack(local_video)
            pc._consumers.append(asyncio.ensure_future(consume_video(track, local_video)))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type='application/json',
        text=json.dumps({
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        }))


pcs = []


async def on_shutdown(app):
    # stop audio / video consumers
    for pc in pcs:
        for c in pc._consumers:
            c.cancel()

    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)

    # --- for worker process --
    executor.shutdown()

# --- detect with YOLO v3 ---
# -- detect in worker --
class DETECT_RESULT(Structure):
    _fields_ = [("left", c_int),
                ("top", c_int),
                ("right", c_int),
                ("bottom", c_int),
                ("cid", c_int)
            ]


def workerDetect(data):
    img = array_to_image(data)
    res = detectObjects(net, meta, img, alphabet)
    return res

# in: im, out:result(name and rect)
def detectObjects(net, meta, im, alp, thresh=.5, hier_thresh=.5, nms=.45):
    #print('------ start detect ----')
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                #res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
                res.append((i, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
                #print("detect " , meta.names[i], dets[j].prob[i])
    res = sorted(res, key=lambda x: -x[1])
    #print(res)

    free_detections(dets, num)
    return res

def onDetected(future):
    #print('--- future on end---')
    #-- get result --
    detect_results = future.result() # <--- wait at here

    # -- clear last results ---
    with workerBusy.get_lock():
        for i in range(max_result_count):
            lastResults[i] = DETECT_RESULT(0, 0, 0, 0, 0)

    i = 0
    for r in detect_results:
        cid = r[0]
        #name = meta.names[cid]
        b = r[2]
        x = b[0]
        y = b[1]
        w = b[2]
        h = b[3]
        left = int(x - w/2)
        top = int(y - h/2)
        right = int(x + w/2)
        bottom = int(y + h/2)

        # -- update last result ---
        with workerBusy.get_lock():
            lastResults[i] = DETECT_RESULT(left, top, right, bottom, cid)

        i = i + 1
        if i >= max_result_count:
            break

    workerBusy.value = 0
    return

# --- init yolo v3 tiny ---
net = load_net("cfg/yolov3-tiny.cfg".encode('ascii'), "yolov3-tiny.weights".encode('ascii'), 0)
meta = load_meta("cfg/coco.data".encode('ascii'))
alphabet = load_alphabet()

# --- status of worker process --
executor = None
#workerBusy = False
#lastDetects = []

workerBusy = Value('i', 0)

max_result_count = 3
lastResults = RawArray(DETECT_RESULT, max_result_count)
for i in range(max_result_count):
    lastResults[i] = DETECT_RESULT(0, 0, 0, 0, 0)

# --- start web app ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WebRTC audio / video / data-channels demo')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port for HTTP server (default: 8080)')
    parser.add_argument('--verbose', '-v', action='count')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # --- executor for worker process --
    # WARN; currentry for 1 PeerConnection
    #executor = ProcessPoolExecutor(1)
    executor = ThreadPoolExecutor(1)


    # --- web app --
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get('/', index)
    app.router.add_get('/client.js', javascript)
    app.router.add_post('/offer', offer)
    web.run_app(app, port=args.port)

