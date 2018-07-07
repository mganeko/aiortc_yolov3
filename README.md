# python3 sample for aiortc with YOLO v3

sample for aiortc with darknet YOLO v3 for Python 3

- aiortc ... WebRTC implementation with Python ([GitHub](https://github.com/jlaine/aiortc))
- YOLO v3 ... object detection network on darknet ([GitHub](https://github.com/pjreddie/darknet/)) 


# prepare

## With Docker



## by hand
- clone and build [aiortc](https://github.com/jlaine/aiortc)
- clone and bulid [darknet](https://github.com/pjreddie/darknet)
- cd darknet
- copy darknet-tiny-label.py to darknet/python
-
- clone [this sample](https://github.com/mganeko/python3_yolov3)
- copy server_yolo.py, index.html to aiortc/examples/server/

# exec

```
$ cd darknet
$ wget https://pjreddie.com/media/files/yolov3-tiny.weights
$ python3 python/darknet-tiny-label.py
```

"detect_result.png" will be created.

