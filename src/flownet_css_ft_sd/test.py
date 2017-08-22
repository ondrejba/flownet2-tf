import argparse
import os, time
from ..net import Mode
from .flownet_css import FlowNetCSS
from scipy.misc import imread, imsave, imresize
import numpy as np

height = 384
width = 512

def main(args):
    # create a new network
    net = FlowNetCSS(mode=Mode.TEST)
    input_a_pl, input_b_pl, pred_flow, sess = \
        net.get_sess_and_flow_op("./checkpoints/FlowNetCSS-ft-sd/flownet-CSS-ft-sd.ckpt-0")

    # frames list
    with open(args.list) as f:
        content = f.readlines()
    meta = [line.strip() for line in content]

    start = time.time()
    for idx, line in enumerate(meta):

        paths = line.split(" ")

        frame1 = imread(paths[0])
        frame2 = imread(paths[1])

        orig_height = frame1.shape[0]
        orig_width = frame1.shape[1]

        frame1 = imresize(frame1, (height, width))
        frame2 = imresize(frame2, (height, width))

        frame1 = frame1[..., [2, 1, 0]]
        frame2 = frame2[..., [2, 1, 0]]

        # Scale from [0, 255] -> [0.0, 1.0] if needed
        if frame1.max() > 1.0:
            frame1 = frame1 / 255.0
        if frame2.max() > 1.0:
            frame2 = frame2 / 255.0

        flw = sess.run(pred_flow, feed_dict={
            input_a_pl: np.expand_dims(frame1, 0),
            input_b_pl: np.expand_dims(frame2, 0)
        })

        if idx != 0 and idx % 100 == 0:
            duration = time.time() - start
            start = time.time()

            print("fps: ", 100 / duration   )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("list")
    parsed = parser.parse_args()
    main(parsed)
