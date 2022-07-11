import argparse
import errno
import os

import cv2
import numpy as np
import pandas as pd


def convert(source, dest_dir, frame_skip, video, grayscale, prepend):
    input = np.loadtxt(source, dtype=int)

    # Make the base directory
    pre_name = source.split('/')[-1].split('.')[0]
    base_path = os.path.join(dest_dir, pre_name)
    try:
        os.mkdir(base_path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            print(error)
            print("Exiting...")
            exit(-1)
    print("Made dir '%s" %base_path)

    # Make the labels directory
    labels_path = os.path.join(base_path, 'labels')
    try:
        os.mkdir(labels_path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            print(error)
            print("Exiting...")
            exit(-1)
    print("Made dir '%s" % labels_path)

    # Make the images directory
    if video:
        images_path = os.path.join(base_path, 'images')
        try:
            os.mkdir(images_path)
        except OSError as error:
            if error.errno != errno.EEXIST:
                print(error)
                print("Exiting...")
                exit(-1)
        print("Made dir '%s" % images_path)

        # Open the video file
        cap = cv2.VideoCapture(video)

    input = input[input[:, 2].argsort()]
    df = pd.DataFrame(input, columns=['object_id', 'object_duration', 'frame','bbox_tl_x','bbox_tl_y','bbox_width','bbox_height','class'])

    current_frame = frame_skip
    f = make_file(labels_path, current_frame, prepend, pre_name)
    for index, row in df.iterrows():
        if row['frame'] == current_frame:
            x_center = (row['bbox_tl_x'] + row['bbox_width']/2) / 1920
            y_center = (row['bbox_tl_y'] + row['bbox_height'] / 2) / 1080
            row_out = [row['class'], x_center, y_center, row['bbox_width']/1920, row['bbox_height']/1080]
            f.write(' '.join(str(e) for e in row_out))
            f.write('\n')
        elif row['frame'] > current_frame:
            if video:
                cap.set(1, current_frame)
                ret, frame = cap.read()
                if grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prepend:
                    cv2.imwrite(os.path.join(images_path, pre_name + '_' + f'{current_frame:08d}' + '.png'), frame)
                else:
                    cv2.imwrite(os.path.join(images_path, f'{current_frame:08d}' + '.png'), frame)

            current_frame = current_frame + frame_skip
            f = make_file(labels_path, current_frame, prepend, pre_name)


    print("Finished!")

def make_file(base_path, frame_num, prepend, pre_name):
    if prepend:
        return open(os.path.join(base_path, pre_name + '_' + f'{frame_num:08d}' + '.txt'), 'w')
    else:
        return open(os.path.join(base_path, f'{frame_num:08d}' + '.txt'), 'w')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='', help='source VIRAT object annotation file')
    parser.add_argument('--dest-dir', type=str, default='',
                        help='destination directory for YOLOv5-formatted annotations')
    parser.add_argument('--frame-skip', default=150, type=int, help='frame interval between output annotations, '
                                                                   'for objects with long durations')
    parser.add_argument('--video', type=str, default='', help='extract and save images from video')
    parser.add_argument('--grayscale', action='store_true', help='convert output images to grayscale')
    parser.add_argument('--prepend', action='store_true', help='prepend filenames with source name')

    opt = parser.parse_args()
    convert(**vars(opt))

