#!/usr/bin/env python

import os
import struct
import sys
import random

from array import array
from os import path

import png

# source: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
def read(dataset = "training", path = "."):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    return lbl, img, size, rows, cols

def write_data(output_dirs, curr_node, label, i, cols, rows):

    output_filename = path.join(output_dirs[curr_node][label], str(i) + ".png")

    filename = path.join(output_filename)
    # print("writing " + filename)
    with open(filename, "wb") as h:
        w = png.Writer(cols, rows, greyscale=True)
        data_i = [
            data[ (i*rows*cols + j*cols) : (i*rows*cols + (j+1)*cols) ]
            for j in range(rows)
        ]
        w.write(h, data_i)

def write_dataset(labels, data, size, rows, cols, output_dir, dataset, num_nodes, percent_uniform,
                  num_datapoints):

    if num_datapoints == -1:
        num_datapoints = len(labels)

    # create output directories
    output_dirs = [
        [path.join(output_dir, str(j), dataset, str(i))
        for i in range(10)]
        for j in range(num_nodes)
    ]
    for node_dir in output_dirs:
        for dir in node_dir:
            if not path.exists(dir):
                os.makedirs(dir)

    ## Write data
    curr_node = 0
    for (i, label) in enumerate(labels):

        # Assign file to node:
        # Split Uniformly, or
        if i < (num_datapoints * percent_uniform):
            write_data(output_dirs, curr_node, label, i, cols, rows)
            curr_node = (curr_node + 1) % num_nodes
        # Assign to partition
        else:
            label_to_node_slope = num_nodes / 10
            start = int(label * label_to_node_slope)
            end = int((label + 1) * label_to_node_slope)
            curr_node = random.randrange(start, end)
            write_data(output_dirs, curr_node, label, i, cols, rows)

        if i >= num_datapoints:
            break

def write_testing_dataset(labels, data, size, rows, cols, output_dir, num_datapoints):

    if num_datapoints == -1:
        num_datapoints = len(labels)

    # print(num_datapoints)

    # create output directories
    output_dirs = [
        path.join(output_dir,f'{i}-test')
        for i in range(10)
    ]
    for dir in output_dirs:
        if not path.exists(dir):
            os.makedirs(dir)

    # write data
    for (i, label) in enumerate(labels):
        output_filename = path.join(output_dirs[label], str(i) + ".png")
        # print("writing " + output_filename)
        with open(output_filename, "wb") as h:
            w = png.Writer(cols, rows, greyscale=True)
            data_i = [
                data[ (i*rows*cols + j*cols) : (i*rows*cols + (j+1)*cols) ]
                for j in range(rows)
            ]
            w.write(h, data_i)

        if i >= num_datapoints:
            break

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("usage: {0} <input_path> <output_path> <num_nodes> <percent_uniform> <num_datapoints>".format(sys.argv[0]))
        sys.exit()

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    num_nodes = int(sys.argv[3])
    percent_uniform = float(sys.argv[4])

    num_datapoints = int(sys.argv[5])

    assert 0 < num_nodes
    assert 0 <= percent_uniform < 1

    # Different Training Dataset per Node
    dataset = "training"

    labels, data, size, rows, cols = read(dataset, input_path)
    write_dataset(labels, data, size, rows, cols,
                    output_path, dataset,
                    num_nodes, percent_uniform, num_datapoints)
    
    dataset = "testing"

    labels, data, size, rows, cols = read(dataset, input_path)
    write_testing_dataset(labels, data, size, rows, cols, output_path,
                          num_datapoints)