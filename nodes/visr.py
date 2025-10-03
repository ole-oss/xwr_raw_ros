#!/usr/bin/env python3

"""Simple visualizer of raw radar data.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
from collections import deque
import openvino.runtime as ov

import rospy
from rospy.numpy_msg import numpy_msg
from xwr_raw_ros.msg import RadarFrame
from xwr_raw_ros.msg import RadarFrameStamped
from xwr_raw_ros.msg import RadarFrameFull
from xwr_raw_ros.msg import RangeOut
from xwr_raw.radar_config import RadarConfig
import xwr_raw.dsp as dsp
import xwr_raw.image_tools as image_tools


def vis_range(args, frame):

        publisher_range = rospy.Publisher('range_data',
                                numpy_msg(RangeOut),
                                queue_size=1)

        if not hasattr(vis_range, 'radar_cubes'):
            vis_range.radar_cubes = deque(maxlen=3)

        if 'xWR68xx' in frame.platform:
            if 'AOP' in frame.platform: #6843AOP
                radar_cube = dsp.reshape_frame(frame, flip_aop_phase=True)

            else: # 6843ISK-ODS
                radar_cube = dsp.reshape_frame(frame, flip_ods_phase=True)

        elif 'xWR18xx' in frame.platform:
            if 'AOP' in frame.platform: #1843AOP
                radar_cube = dsp.reshape_frame(frame)

            else: #1843
                radar_cube = dsp.reshape_frame(frame)
        else:
            raise ValueError('Unknown radar type')

        vis_range.radar_cubes.append(radar_cube)
        if len(vis_range.radar_cubes) < vis_range.radar_cubes.maxlen:
            return
        radar_cube = np.concatenate(vis_range.radar_cubes, axis=0)

        if 'xWR68xx' in frame.platform:
            if 'AOP' in frame.platform: #6843AOP

                radar_cube = np.stack([radar_cube[:,11,:],
                                       radar_cube[:,9,:],
                                       radar_cube[:,7,:],
                                       radar_cube[:,5,:]], axis=1)

                # radar_cube = dsp._tdm(radar_cube, 2, 2)
            else: # 6843ISK-ODS

                # radar_cube = np.stack([radar_cube[:,0,:],
                #                        radar_cube[:,3,:],
                #                        radar_cube[:,4,:],
                #                        radar_cube[:,7,:]], axis=1)

                radar_cube = np.stack([radar_cube[:,4,:],
                                       radar_cube[:,5,:],
                                       radar_cube[:,8,:],
                                       radar_cube[:,9,:]], axis=1)

                radar_cube = dsp._tdm(radar_cube, 2, 2)

        elif 'xWR18xx' in frame.platform:
            if 'AOP' in frame.platform: #1843AOP

                radar_cube = radar_cube[:,:4,:] \
                           + radar_cube[:,4:8,:] \
                           + radar_cube[:,8:12,:]

                # radar_cube = dsp._tdm(radar_cube, 3, 4)
                # radar_cube = dsp._tdm(radar_cube, 2, 2)
                # radar_cube = dsp._tdm(radar_cube, 4, 1)

            else: #1843

                radar_cube = radar_cube[:,:8,:]

                radar_cube = dsp._tdm(radar_cube, 2, 4)

        else:
            raise ValueError('Unknown radar type')


        # print(dsp.compute_altitude(radar_cube,
        #                            frame.range_max/frame.shape[2],
        #                            frame.range_bias))


        """
        radar_cube: chirp x rx x adc
        Returns a centered (two-sided) range profile in dB.
        """
        # Sum across RX antennas
        sum_rx = np.sum(radar_cube, axis=1)  # shape: chirps x adc
        range_response = np.fft.fft(sum_rx, axis=1)
        range_response = np.fft.fftshift(range_response, axes=1)
        range_response_1d = np.sum(np.abs(range_response), axis=0)
        # return 20 * np.log10(range_response_1d + 1e-12)

        img = image_tools.waveform_disp(20 * np.log10(range_response_1d + 1e-12))

        msg = RangeOut()
        npmsgData = 20 * np.log10(range_response_1d + 1e-12)
        msg.data=npmsgData.astype(np.int16).tolist()
        # rospy.loginfo(str(msg.data))
        publisher_range.publish(msg)


        # # if frame.adc_output_fmt > 0:
        # #     range_plot = dsp.compute_range_azimuth_capon(radar_cube, 1, 90)
        # # else:
        # #     range_plot = dsp.compute_range_azimuth_capon_real(radar_cube, 1, 90)

        # # img = image_tools.polar2cartesian(
        # #     range_plot,
        # #     np.linspace(
        # #         frame.range_bias,
        # #         frame.range_max,
        # #         range_plot.shape[0]
        # #     ),
        # #     np.linspace(
        # #         np.deg2rad(90),
        # #         -np.deg2rad(90),
        # #         range_plot.shape[1]
        # #     ),
        # #     np.linspace(
        # #         0.,
        # #         frame.range_max,
        # #         range_plot.shape[0]
        # #     ),
        # #     np.arange(
        # #         -frame.range_max*np.sin(np.deg2rad(90)),
        # #         frame.range_max*np.sin(np.deg2rad(90)),
        # #         frame.range_max/range_plot.shape[0]
        # #     )
        # # )
        # # # img = range_plot

        # # img = np.rot90(img)
        # # img = image_tools.normalize_and_color(img, 
        # #                                       min_val=0, max_val=18.0, 
        # #                                       cmap=cv2.COLORMAP_JET)

        cv2.namedWindow('range_plot', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('range_plot', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('visualizer')

    subscriber_radar = rospy.Subscriber('radar_data',
                                        numpy_msg(RadarFrameFull),
                                        lambda frame: vis_range(args, frame),
                                        queue_size=1)

    rospy.spin()

    

