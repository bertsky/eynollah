import numpy as np

from eynollah.utils.separate_lines import return_deskew_slop

def test_deskew_landscape_main_page():
    # landscape page (width > height) used to crash with
    # "TypeError: 'int' object is not iterable" (#223)
    img = np.zeros((80, 160))
    for y in (20, 40, 60):
        img[y:y + 3, 10:150] = 1
    angle = return_deskew_slop(img, 2, n_tot_angles=10, main_page=True)
    assert np.isfinite(angle)
