# modified from https://github.com/phamquiluan/jdeskew/blob/master/jdeskew/estimator.py
# (to better control search range)

"""Skew Estimator."""
import cv2
import numpy as np
from typing import Optional
from functools import partial

from .shm import share_ndarray, wrap_ndarray_shared

def _ensure_gray(image: np.ndarray) -> np.ndarray:
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        pass
    return image


def _ensure_optimal_square(image: np.ndarray) -> np.ndarray:
    assert image is not None, image
    nw = nh = cv2.getOptimalDFTSize(max(image.shape[:2]))
    output_image = cv2.copyMakeBorder(
        src=image,
        top=0,
        bottom=nh - image.shape[0],
        left=0,
        right=nw - image.shape[1],
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )
    return output_image


def _get_fft_magnitude(image: np.ndarray) -> np.ndarray:
    gray = _ensure_gray(image)
    #dims = np.array(gray.shape[:2])
    #marg = (dims * [[0.1, 0.9], [0.1, 0.9]]).astype(int)
    #gray = gray[marg[0,0]:marg[0,1], marg[1,0]:marg[1,1]]
    opt_gray = _ensure_optimal_square(gray)

    # thresh
    opt_gray = cv2.adaptiveThreshold(
        ~opt_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10
    )

    # perform fft - using fft2 to ensure square output
    dft = np.fft.fft2(opt_gray)
    shifted_dft = np.fft.fftshift(dft)

    # get the magnitude (module)
    magnitude = np.abs(shifted_dft)
    return magnitude

@wrap_ndarray_shared(kw='m')
def _sum_radial_projection(
        t: float,
        m: np.ndarray = None) -> float:
    assert m.shape[0] == m.shape[1]
    r = c = m.shape[0] // 2
    x = np.arange(0, r)
    y = c + np.int32(x * np.cos(t))
    x = c + np.int32(-1 * x * np.sin(t))
    # Use boolean indexing for faster computation
    valid_indices = (y >= 0) & (y < m.shape[0]) & (x >= 0) & (x < m.shape[1])
    return np.sum(m[y[valid_indices], x[valid_indices]])

def _get_angle_radial_projection(
        m: np.ndarray,
        angles: Optional[np.ndarray] = None,
        map=None) -> float:
    """Get angle via radial projection.

    Arguments:
    ------------
    angle_max : float
    num : int
      number of angles to generate between 1 degree
    """
    assert m.shape[0] == m.shape[1]
    r = c = m.shape[0] // 2

    if angles is None:
        angle_max = 15.0
        num = 20
        tr = np.linspace(-1 * angle_max, angle_max, int(angle_max * num * 2)) / 180 * np.pi
    else:
        tr = angles / 180 * np.pi

    if map is None:
        # Pre-allocate array for better performance
        li = np.zeros_like(tr)
        for i, t in enumerate(tr):
            li[i] = _sum_radial_projection.__wrapped__(t, m=m)
    else:
        with share_ndarray(m) as m_shared:
            li = list(map(partial(_sum_radial_projection, m=m_shared), tr))

    a_max = np.argmax(li)
    #a_min = np.argmin(li)
    if a_max == 0:
        if li[1] == li[0]:
            return 0.0, -1
        a_nxt = 1
    elif a_max + 1 >= len(tr):
        a_nxt = a_max - 1
    elif li[a_max - 1] > li[a_max + 1]:
        a_nxt = a_max - 1
    else:
        a_nxt = a_max + 1
    #d = li[a_max] - li[a_nxt]
    #d = li[a_max] - li[a_min]
    d = li[a_max]
    a = tr[a_max] / np.pi * 180
    return float(a), float(d)


def get_angle(
    image: np.ndarray,
    vertical_image_shape: Optional[int] = None, #3072,
    angles: Optional[np.ndarray] = None,
    map=None
) -> float:
    """Getting angle from a given document image.

    Args:
        image: Input image as numpy array
        vertical_image_shape: Optional resize height for preprocessing
        angles: angles to search for

    Returns:
        float: Estimated skew angle in degrees
    """
    assert isinstance(image, np.ndarray), image

    # if vertical_image_shape is None:
    #     vertical_image_shape = 512

    # resize
    if vertical_image_shape is not None:
        ratio = vertical_image_shape / image.shape[0]
        image = cv2.resize(image, None, fx=ratio, fy=ratio)

    m = _get_fft_magnitude(image)
    a, d = _get_angle_radial_projection(m, angles=angles, map=map)
    if d < 0:
        return a, d
    # extra precision
    a2, d2 = _get_angle_radial_projection(m, angles=np.linspace(a - 1.1, a + 1.1, 20), map=map)
    if d2 > d:
        return a2, d2
    return a, d
