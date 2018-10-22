from constant import DATAPATH, INPUT_IMG, OUTPUTPATH

import os
import cv2
import numpy as np
from pdftabextract import imgproc
from pdftabextract.clustering import find_clusters_1d_break_dist
from pdftabextract.clustering import calc_cluster_centers_1d


MIN_COL_WIDTH = 30


def line_extract(image_file):
    '''
    extrace line in the image, get the vertical and horizontal line through all along the image
    :param image_file: file_name in path DATAPATH
    :return: {'verticals': list of vertical lines, 'horizontals': list of horizontal lines}
    '''
    input_img_path = os.path.join(DATAPATH, image_file)
    save_img_file = os.path.join(OUTPUTPATH, '%s_lines.png' % image_file)

    image = cv2.imread(input_img_path)
    iproc_obj = imgproc.ImageProc(input_img_path)

    width, height, _ = image.shape

    # detect the lines
    iproc_obj.detect_lines(canny_kernel_size=3, canny_low_thresh=50, canny_high_thresh=150,
                           hough_rho_res=1, hough_theta_res=np.pi/500, hough_votes_thresh=round(0.2 * iproc_obj.img_w))

    def line_cluster(direction):
        '''
        detect_lines get several lines for every line, so make clusters.
        And Paint the result on origin image.
        :param direction: 'vertical' or 'horizontal'
        :return lines: list of location of line
        '''
        assert direction == 'vertical' or direction == 'horizontal'

        mode_manager = {'vertical': {'imgproc.DIRECTION': imgproc.DIRECTION_VERTICAL, 'paint_color': (0, 0, 255)},
                        'horizontal': {'imgproc.DIRECTION': imgproc.DIRECTION_HORIZONTAL, 'paint_color': (0, 255, 0)}
                        }

        clusters = iproc_obj.find_clusters(mode_manager[direction]['imgproc.DIRECTION'], find_clusters_1d_break_dist,
                                           dist_thresh=MIN_COL_WIDTH / 2)
        lines = np.array(calc_cluster_centers_1d(clusters))

        # paint on orgin image
        for v in lines:
            if direction == 'vertical': start_pot, end_pot = (int(v), 1), (int(v), width)
            if direction == 'horizontal': start_pot, end_pot = (1, int(v)), (height, int(v))
            cv2.line(image, start_pot, end_pot, mode_manager[direction]['paint_color'], 1)

        return lines


    verticals = line_cluster('vertical')
    horizontals = line_cluster('horizontal')

    cv2.imwrite(save_img_file, image)
    return {'verticals': verticals, 'horizontals': horizontals}


def main():
    INPUT_IMG = 'DrillDrawingThroughgdo.png'
    print(line_extract(INPUT_IMG))


if __name__ == '__main__':
    main()
