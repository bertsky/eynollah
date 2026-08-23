import numpy as np
import cv2
from shapely.geometry import Polygon, LineString
from shapely.affinity import rotate as rotate_polygon
from shapely.ops import split as split_polygon
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from .contour import (
    contour2polygon,
    polygon2contour,
    return_contours_of_interested_region,
)
from .resize import resize_image
from .rotate import rotate_image

# from matplotlib import pyplot as plt
# from shapely.plotting import plot_polygon

MIN_THICKNESS_MAINTEXT_PRCT_HEIGHT = 14
"""minimum vertical extent of the text mask (in percent of image height)

(pages with smaller share of text will not get marginalia)"""

MIN_DIST_GAPS = 40 # 20
"""minimum horizontal distance between neighbouring gaps to be candidates for margin points

(Gaps in the text mask projection are primary candidates.)"""

MAX_THICKNESS_GAPS_FRACT_MAIN = 17. # 20. # 30.
"""maximum vertical height of gaps (relative to the maximum height) to be candidates for margin points

(Gaps in the text mask projection are primary candidates.)"""

MIN_THICKNESS_STEP_FRACT_MAIN = 0.05
"""minimum vertical height of margin plateaus (relative to the main average) to be candidates for margin points

(Steps in the text mask projection are fallback candidates.)"""

MAX_THICKNESS_STEP_FRACT_MAIN = 0.5
"""maximum vertical height of margin plateaus (relative to the main average) to be candidates for margin points

(Steps in the text mask projection are fallback candidates.)"""


def get_marginals(num_col, slope_deskew,
                  early_layout,
                  textline_mask,
                  kernel=None,
                  label_text=1,
                  label_marg=4,
                  label_tabs=10,
):
    """
    Detect left and right margins, apply to given segmentation labelling.

    Find horizontal gaps in the (deskewed) text mask (as minima in its
    vertical projection curve).

    If there are multiple candidates on one side, then pick the one
    with highest prominence (drop towards neighbour) and deepest gap
    (lowest minimum).

    If there are no candidates on one side, then try finding steps
    in the text mask (as jumps in its vertical projection curve).
    Pick the steepest jumps that introduce a suitable plateau.

    Constrain the ranges of left and right points depending on the
    given number of columns:
    1. for one-column pages, the left / right positions can be between
       the first / last occurrance of text and the middle of both
    2. for two-column pages, similarly, but also one third towards
       the left / right from the middle.

    (Even one-column pages are allowed to have marginalia on both sides.)

    Use the retrieved points on the left and right side to define margins,
    respectively. Rotate the margins and main masks back to the original
    (skewed) text mask.

    Then extract all contours in the text mask of the region segmentation
    and of the line segmentation. For each contour, if it falls completely
    within either margin, then mark it for marginalia. If it intersects
    either margin, then split it: For each resulting section, if it falls
    completely in the margin, then mark it. But ignore any parts which are
    mostly vertical, when the contour itself was not (to counter imprecision
    of the straight cut).

    Finally, fill all marked contours and partial contours with the marginalia
    label.

    Also, for the text mask of the line segmentation, cut along the left / right
    margin lines (by drawing a line with the background label), to ensure that
    text lines will also be assigned separately.

    Return without changes if
    - there is not enough text (i.e. the maximum of the vertical projection
      of the text mask is too small)
    - there are no sufficient peaks
    - there would be no remaining main text
    """
    kernel = np.ones((2, 2), dtype=np.uint8)
    kernel_hor = np.ones((1, 2), dtype=np.uint8)

    text_mask = ((early_layout == label_text) |
                 (early_layout == label_tabs)).astype(np.uint8)
    text_mask_d = rotate_image(text_mask, slope_deskew)
    height, width = text_mask_d.shape

    # plt.figure("text mask")
    # plt.subplot(1, 3, 1, title="original text mask")
    # plt.imshow(text_mask_d)
    if height <= 1500:
        pass
    elif 1500 < height <= 1800:
        text_mask_d = resize_image(text_mask_d, int(height / 1.5), width)
        text_mask_d = cv2.erode(text_mask_d, kernel, iterations=3)
        # rs: and back to original size
        text_mask_d = resize_image(text_mask_d, height, width)
    else:
        text_mask_d = resize_image(text_mask_d, int(height / 1.8), width)
        text_mask_d = cv2.erode(text_mask_d, kernel, iterations=5)
        # rs: and back to original size
        text_mask_d = resize_image(text_mask_d, height, width)
    # plt.subplot(1, 3, 2, title="eroded text mask")
    # plt.imshow(text_mask_d)

    text_mask_d = cv2.erode(text_mask_d, kernel_hor, iterations=4)
    # plt.subplot(1, 3, 3, title="horizontally eroded")
    # plt.imshow(text_mask_d)
    # plt.show()
    text_mask_d_y = text_mask_d.sum(axis=0)

    max_text_thickness = text_mask_d_y.max()
    max_text_thickness_percent = 100. * max_text_thickness / height
    min_text_thickness = max_text_thickness / MAX_THICKNESS_GAPS_FRACT_MAIN

    if max_text_thickness_percent < MIN_THICKNESS_MAINTEXT_PRCT_HEIGHT:
        # plt.figure("not enough text")
        # ax1 = plt.subplot(2, 2, 1, title="text_mask_d")
        # ax1.imshow(text_mask_d, aspect='auto')
        # ax2 = plt.subplot(2, 2, 3, title="text_mask_d_y", sharex=ax1)
        # ax2.plot(list(range(width)), text_mask_d_y)
        # ax2.hlines(int(0.14 * height), 0, width,
        #            label='max_text_thickness=14%', colors='r')
        # ax2.hlines([min_text_thickness], 0, width,
        #            label='min_text_thickness', colors='g')
        # ax2.scatter([np.argmax(text_mask_d_y)],
        #             [text_mask_d_y.max()], color='r',
        #             label='max = %d%%' % max_text_thickness_percent)
        # ax2.legend()
        # ax1 = plt.subplot(2, 2, 4, title="early layout")
        # ax1.imshow(early_layout, aspect='auto')
        # plt.legend()
        # plt.show()
        return

    region_sum_0 = gaussian_filter1d(text_mask_d_y, 3)
    first_nonzero = region_sum_0.nonzero()[0][0] # outer left
    last_nonzero = region_sum_0.nonzero()[0][-1] # outer right
    mid_point = (last_nonzero + first_nonzero) // 2
    one_third_r = (last_nonzero - mid_point) / 3.0
    one_third_l = (mid_point - first_nonzero) / 3.0

    # find minima (horizontal gaps)
    peaks, props = find_peaks(max_text_thickness - region_sum_0,#text_mask_d_y,
                              # constrain thickness lower bound (min_text_thickness)
                              #height=0.5 * max_text_thickness,
                              height=max_text_thickness - min_text_thickness,
                              # constrain the prominence towards neighbours
                              prominence=0.5 * min_text_thickness,
                              # constrain the distance at least 2 characters at 12pt
                              distance=MIN_DIST_GAPS)
    
    # plt.figure("peaks")
    # ax1 = plt.subplot(2, 2, 1, title="text_mask_d")
    # ax1.imshow(text_mask_d, aspect='auto')
    # ax2 = plt.subplot(2, 2, 3, title="text_mask_d_y", sharex=ax1)
    # ax2.plot(list(range(width)), text_mask_d_y, label='unsmoothed', color='b')
    # ax2.plot(list(range(width)), region_sum_0, label='smoothed', color='m')
    # ax2.hlines(int(0.14 * height), 0, width,
    #            label='max_text_thickness=14%', colors='r')
    # ax2.hlines([min_text_thickness], 0, width,
    #            label='min_text_thickness', colors='m')
    # ax2.vlines([first_nonzero], 0, height, label='first_nonzero', colors='r')
    # ax2.vlines([last_nonzero], 0, height, label='last_nonzero', colors='r')
    # if num_col == 2:
    #     ax2.vlines([mid_point], 0, height, label='mid_point', colors='r')
    # ax2.scatter([np.argmax(text_mask_d_y)],
    #             [text_mask_d_y.max()], color='r',
    #             label='max = %d%%' % max_text_thickness_percent)
    # ax2.scatter(peaks, region_sum_0[peaks], label='peaks', color='m')
    # #ax2.scatter(peaks, props['prominences'], label='prominences')
    # ax1 = plt.subplot(2, 2, 4, title="early layout")
    # ax1.imshow(early_layout, aspect='auto')
    # plt.legend()

    # also calculate the product of prominence and height (for final selection)
    scores = np.zeros(peaks.max(initial=width) + 1)
    scores[peaks] = props['prominences'] * props['peak_heights']

    peaks = peaks[(peaks > first_nonzero) & (peaks < last_nonzero)]

    if num_col == 1:
        peaks_r = peaks[peaks > mid_point]
        peaks_l = peaks[peaks < mid_point]
    elif num_col == 2:
        peaks_r = peaks[peaks > mid_point + one_third_r]
        peaks_l = peaks[peaks < mid_point - one_third_l]
    else:
        # should not happen, anyway
        return

    # if there are no valid peaks (i.e. gaps) on either side,
    # because marginalia are very close to main,
    # then look at positive/negative peaks of derivative on left/right side,
    # to identify smaller plateaus of sufficient thickness:
    if len(peaks_l) == 0:
        region_sum_1 = np.diff(region_sum_0[:mid_point+1].astype(int))
        peaks2, _ = find_peaks(region_sum_1,
                               # constrain slope lower bound
                               height=0.04 * max_text_thickness,
                               # constrain the distance at least 2 characters at 12pt
                               distance=MIN_DIST_GAPS)
        # ax2.plot(list(range(mid_point)), region_sum_1, label='derivative', color='g')
        # ax2.scatter(peaks2, region_sum_1[peaks2], label='peaks2', color='g')
        # ax2.hlines([0.04 * max_text_thickness], 0, width,
        #            label='min2', colors='lightgreen')
        if num_col == 1:
            peaks2_l = peaks2[peaks2 < mid_point]
        else:
            peaks2_l = peaks2[peaks2 < mid_point - one_third_l]
        # search from right (widest) to left
        for peak2 in reversed(peaks2_l):
            if peak2 < first_nonzero + MIN_DIST_GAPS:
                continue
            # non-zero, non-max plateau
            if (MIN_THICKNESS_STEP_FRACT_MAIN
                < (np.mean(region_sum_0[first_nonzero: peak2]) / 
                   np.mean(region_sum_0[peak2: mid_point])) <
                MAX_THICKNESS_STEP_FRACT_MAIN):
                peaks_l = [peak2]
                # ax2.vlines([peak2], 0, height, label='peak2', colors='y')
                break
            # TODO: try another criterion: deviation in average textline heights
    if len(peaks_r) == 0:
        region_sum_1 = -np.diff(region_sum_0[mid_point-1:].astype(int))
        peaks2, _ = find_peaks(region_sum_1,
                               # constrain slope lower bound
                               height=0.04 * max_text_thickness,
                               # constrain the distance at least 2 characters at 12pt
                               distance=MIN_DIST_GAPS)
        # ax2.plot(np.arange(mid_point, width), region_sum_1, label='derivative', color='g')
        # ax2.scatter(peaks2 + mid_point, region_sum_1[peaks2], label='peaks2', color='g')
        # ax2.hlines([0.04 * max_text_thickness], 0, width,
        #            label='min2', colors='lightgreen')
        peaks2 += mid_point
        if num_col == 1:
            peaks2_r = peaks2[peaks2 > mid_point]
        else:
            peaks2_r = peaks2[peaks2 > mid_point + one_third_r]
        # search from left (widest) to right
        for peak2 in peaks2_r:
            if peak2 > last_nonzero - MIN_DIST_GAPS:
                continue
            # non-zero, non-max plateau
            if (MIN_THICKNESS_STEP_FRACT_MAIN
                < (np.mean(region_sum_0[peak2: last_nonzero]) /
                   np.mean(region_sum_0[mid_point: peak2])) <
                MAX_THICKNESS_STEP_FRACT_MAIN):
                peaks_r = [peak2]
                # ax2.vlines([peak2], 0, height, label='peak2', colors='y')
                break
    # ax2.legend()
    # plt.show()

    if len(peaks_l) == 0:
        if len(peaks_r) == 0:
            # plt.figure("no left or right peaks")
            # ax1 = plt.subplot(2, 1, 1, title='text_mask_d (deskewed text+sep mask)')
            # ax1.imshow(text_mask_d, aspect='auto')
            # ax1.vlines([first_nonzero], 0, height, label='first_nonzero', colors='r')
            # ax1.vlines([last_nonzero], 0, height, label='last_nonzero', colors='r')
            # ax1.vlines(peaks_l, 0, height, label='peaks_l', colors='orange')
            # ax1.vlines(peaks_r, 0, height, label='peaks_r', colors='orange')
            # ax1.legend()
            # ax2 = plt.subplot(2, 1, 2, title='text_mask_d_y (smoothed)', sharex=ax1)
            # ax2.plot(list(range(width)), region_sum_0)
            # ax2.hlines(min_text_thickness, 0, width, colors='g',
            #            label='min_text_thickness=%d' % min_text_thickness)
            # ax2.scatter(peaks_orig, region_sum_0[peaks_orig], label='peaks')
            # ax2.legend()
            # plt.show()
            return
        point_r = peaks_r[np.argmax(scores[peaks_r])]
        #point_l = first_nonzero
        point_l = 0
    elif len(peaks_r) == 0:
        point_l = peaks_l[np.argmax(scores[peaks_l])]
        #point_r = last_nonzero
        point_r = width - 1
    else:
        best_l = np.argmax(scores[peaks_l])
        best_r = np.argmax(scores[peaks_r])
        point_l = peaks_l[best_l]
        point_r = peaks_r[best_r]
        if scores[best_l] < 0.1 * scores[best_r]:
            point_l = 0
            #point_l = first_nonzero
        if scores[best_r] < 0.1 * scores[best_l]:
            point_r = 0
            #point_r = last_nonzero

    # plt.figure("final peaks")
    # ax1 = plt.subplot(2, 2, 1)
    # ax1.title.set_text('text_mask_d (deskewed text+table mask)')
    # ax1.imshow(text_mask_d)
    # ax1.vlines(peaks_l, 0, height, label='peaks_l', colors='b')
    # ax1.vlines(peaks_r, 0, height, label='peaks_r', colors='b')
    # ax1.vlines([first_nonzero], 0, height, label='first_nonzero', colors='g')
    # ax1.vlines([last_nonzero], 0, height, label='last_nonzero', colors='g')
    # ax1.vlines([point_l], 0, height, label='point_l', colors='r')
    # ax1.vlines([point_r], 0, height, label='point_r', colors='r')
    # ax2 = plt.subplot(2, 2, 2, title='main_mask_d (deskewed main mask)', sharey=ax1)
    # ax3 = plt.subplot(2, 2, 3, title='text_mask_d_y (projection for minima)', sharex=ax1)
    # ax3.plot(list(range(width)), text_mask_d_y)
    # ax3.set_aspect('auto')
    # ax3.vlines([point_l], 0, height, label='point_l', colors='r')
    # ax3.vlines([point_r], 0, height, label='point_r', colors='r')
    # ax4 = plt.subplot(2, 2, 4, title='early_layout (undeskewed labels)')
    # ax4.imshow(early_layout)
    # plt.legend()
    # plt.show()

    # rotate back (into undeskewed/original shape as early_layout input):
    marg_l = Polygon([[point_l, 0], [point_l, height],
                      [0, height], [0, 0]])
    marg_r = Polygon([[point_r, 0], [point_r, height],
                      [width, height], [width, 0]])
    main = Polygon([[point_l, 0], [point_l, height],
                    [point_r, height], [point_r, 0]])
    # plt.imshow(text_mask_d)
    # plot_polygon(marg_l, color='yellow')
    # plot_polygon(marg_r, color='red')
    # plot_polygon(main, color='magenta')
    # plt.show()
    marg_l = rotate_polygon(marg_l, slope_deskew, origin=(0.5 * width, 0.5 * height))
    marg_r = rotate_polygon(marg_r, slope_deskew, origin=(0.5 * width, 0.5 * height))
    main = rotate_polygon(main, slope_deskew, origin=(0.5 * width, 0.5 * height))
    # plot_polygon(marg_l, color='yellow')
    # plot_polygon(marg_r, color='red')
    # plot_polygon(main, color='magenta')
    # plt.show()
    line_l = LineString(marg_l.exterior.coords[:2])
    line_r = LineString(marg_r.exterior.coords[:2])
    marg_l3 = marg_l.buffer(3)
    marg_r3 = marg_r.buffer(3)
    line_l3 = line_l.offset_curve(5)
    line_r3 = line_r.offset_curve(-5)

    # re-assign (marg/main), cutting through existing contours if necessary
    min_area_text = 0.00001
    text_contours = return_contours_of_interested_region(early_layout, label_text, min_area_text)
    marg_contours = []
    for cont in text_contours:
        poly = contour2polygon(cont)
        if not poly.area:
            continue # fixme: warn
        minx, miny, maxx, maxy = poly.bounds
        aspect = (maxy - miny) / (maxx - minx)
        if poly.within(marg_l3) or poly.within(marg_r3):
            marg_contours.append(cont)
        elif poly.within(main):
            continue
        elif poly.intersects(marg_l) or poly.intersects(marg_r):
            # partial: split, but ignore marg parts overly vertical
            # plot_polygon(main, color='magenta')
            inter = poly.intersection(marg_l)
            iminx, iminy, imaxx, imaxy = inter.bounds
            iaspect = (imaxy - iminy) / (imaxx - iminx)
            # ignore if intersection is almost vertical
            if not (iaspect > 4 and iaspect > 8 * aspect):
                for geom in split_polygon(poly, line_l3).geoms:
                    gminx, gminy, gmaxx, gmaxy = geom.bounds
                    gaspect = (gmaxy - gminy) / (gmaxx - gminx)
                    if gaspect > 4 and gaspect > 8 * aspect:
                        continue
                    if not geom.within(marg_l3):
                        continue
                    # plot_polygon(geom, color='yellow')
                    marg_contours.append(polygon2contour(geom))
            inter = poly.intersection(marg_r)
            iminx, iminy, imaxx, imaxy = inter.bounds
            iaspect = (imaxy - iminy) / (imaxx - iminx)
            # ignore if intersection is almost vertical
            if not (iaspect > 4 and iaspect > 8 * aspect):
                for geom in split_polygon(poly, line_r3).geoms:
                    gminx, gminy, gmaxx, gmaxy = geom.bounds
                    gaspect = (gmaxy - gminy) / (gmaxx - gminx)
                    if gaspect > 4 and gaspect > 8 * aspect:
                        continue
                    if not geom.within(marg_r3):
                        continue
                    # plot_polygon(geom, color='red')
                    marg_contours.append(polygon2contour(geom))
            # plt.show()
        else:
            assert False

    # write marginals to segmentation
    early_layout = cv2.fillPoly(early_layout, pts=marg_contours, color=label_marg)

    # also cut textlines
    # plt.subplot(1, 2, 1, title='textline mask')
    # plt.imshow(textline_mask)
    line_l = np.stack(line_l.xy).astype(int)
    line_r = np.stack(line_r.xy).astype(int)
    textline_mask = cv2.line(textline_mask, line_l[:, 0], line_l[:, 1], 0, 1)
    textline_mask = cv2.line(textline_mask, line_r[:, 0], line_r[:, 1], 0, 1)
    # plt.subplot(1, 2, 2, title='textline mask (split)')
    # plt.imshow(textline_mask)
    # plt.show()

    # if there was no main text, then relabel marginalia as main
    if not np.any(early_layout == label_text):
        early_layout[early_layout == label_marg] = label_text
