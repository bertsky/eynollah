# pylint: disable=no-member,invalid-name,line-too-long,missing-function-docstring,missing-class-docstring,too-many-branches
# pylint: disable=too-many-locals,wrong-import-position,too-many-lines,too-many-statements,chained-comparison,fixme,broad-except,c-extension-no-member
# pylint: disable=too-many-public-methods,too-many-arguments,too-many-instance-attributes,too-many-public-methods,
# pylint: disable=consider-using-enumerate
"""
document layout analysis (segmentation) with output in PAGE-XML
"""

import math
import os
import sys
import time
import warnings
from pathlib import Path
from multiprocessing import Process, Queue, cpu_count
import gc
from ocrd_utils import getLogger
import cv2
import numpy as np
from transformers import TrOCRProcessor
from PIL import Image
import torch
from difflib import SequenceMatcher as sq
from transformers import VisionEncoderDecoderModel
from numba import cuda 
import copy
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import load_model
sys.stderr = stderr
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
# use tf1 compatibility for keras backend
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras import layers

from .utils.contour import (
    filter_contours_area_of_image,
    filter_contours_area_of_image_tables,
    find_contours_mean_y_diff,
    find_new_features_of_contours,
    find_features_of_contours,
    get_text_region_boxes_by_given_contours,
    get_textregion_contours_in_org_image,
    get_textregion_contours_in_org_image_light,
    return_contours_of_image,
    return_contours_of_interested_region,
    return_contours_of_interested_region_by_min_size,
    return_contours_of_interested_textline,
    return_parent_contours,
)
from .utils.rotate import (
    rotate_image,
    rotation_not_90_func,
    rotation_not_90_func_full_layout)
from .utils.separate_lines import (
    textline_contours_postprocessing,
    separate_lines_new2,
    return_deskew_slop)
from .utils.drop_capitals import (
    adhere_drop_capital_region_into_corresponding_textline,
    filter_small_drop_capitals_from_no_patch_layout)
from .utils.marginals import get_marginals
from .utils.resize import resize_image
from .utils import (
    boosting_headers_by_longshot_region_segmentation,
    crop_image_inside_box,
    find_num_col,
    otsu_copy_binary,
    put_drop_out_from_only_drop_model,
    putt_bb_of_drop_capitals_of_model_in_patches_in_layout,
    check_any_text_region_in_model_one_is_main_or_header,
    check_any_text_region_in_model_one_is_main_or_header_light,
    small_textlines_to_parent_adherence2,
    order_of_regions,
    find_number_of_columns_in_document,
    return_boxes_of_images_by_order_of_reading_new)
from .utils.pil_cv2 import check_dpi, pil2cv
from .utils.xml import order_and_id_of_texts
from .plot import EynollahPlotter
from .writer import EynollahXmlWriter

MIN_AREA_REGION = 0.000001
SLOPE_THRESHOLD = 0.13
RATIO_OF_TWO_MODEL_THRESHOLD = 95.50 #98.45:
DPI_THRESHOLD = 298
MAX_SLOPE = 999
KERNEL = np.ones((5, 5), np.uint8)

projection_dim = 64
patch_size = 1
num_patches =21*21#14*14#28*28#14*14#28*28


class Patches(layers.Layer):
    def __init__(self, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })
        return config
    
    
class PatchEncoder(layers.Layer):
    def __init__(self, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection': self.projection,
            'position_embedding': self.position_embedding,
        })
        return config

class Eynollah:
    def __init__(
        self,
        dir_models,
        image_filename=None,
        image_pil=None,
        image_filename_stem=None,
        dir_out=None,
        dir_in=None,
        dir_of_cropped_images=None,
        extract_only_images=False,
        dir_of_layout=None,
        dir_of_deskewed=None,
        dir_of_all=None,
        dir_save_page=None,
        enable_plotting=False,
        allow_enhancement=False,
        curved_line=False,
        textline_light=False,
        full_layout=False,
        tables=False,
        right2left=False,
        input_binary=False,
        allow_scaling=False,
        headers_off=False,
        light_version=False,
        ignore_page_extraction=False,
        reading_order_machine_based=False,
        do_ocr=False,
        num_col_upper=None,
        num_col_lower=None,
        skip_layout_and_reading_order = False,
        override_dpi=None,
        logger=None,
        pcgts=None,
    ):
        self.light_version = light_version
        if not dir_in:
            if image_pil:
                self._imgs = self._cache_images(image_pil=image_pil)
            else:
                self._imgs = self._cache_images(image_filename=image_filename)
            if override_dpi:
                self.dpi = override_dpi
            self.image_filename = image_filename
        self.dir_out = dir_out
        self.dir_in = dir_in
        self.dir_of_all = dir_of_all
        self.dir_save_page = dir_save_page
        self.reading_order_machine_based = reading_order_machine_based
        self.dir_of_deskewed = dir_of_deskewed
        self.dir_of_deskewed =  dir_of_deskewed
        self.dir_of_cropped_images=dir_of_cropped_images
        self.dir_of_layout=dir_of_layout
        self.enable_plotting = enable_plotting
        self.allow_enhancement = allow_enhancement
        self.curved_line = curved_line
        self.textline_light = textline_light
        self.full_layout = full_layout
        self.tables = tables
        self.right2left = right2left
        self.input_binary = input_binary
        self.allow_scaling = allow_scaling
        self.headers_off = headers_off
        self.light_version = light_version
        self.extract_only_images = extract_only_images
        self.ignore_page_extraction = ignore_page_extraction
        self.skip_layout_and_reading_order = skip_layout_and_reading_order
        self.ocr = do_ocr
        if num_col_upper:
            self.num_col_upper = int(num_col_upper)
        else:
            self.num_col_upper = num_col_upper
        if num_col_lower:
            self.num_col_lower = int(num_col_lower)
        else:
            self.num_col_lower = num_col_lower
        self.pcgts = pcgts
        if not dir_in:
            self.plotter = None if not enable_plotting else EynollahPlotter(
                dir_out=self.dir_out,
                dir_of_all=dir_of_all,
                dir_save_page=dir_save_page,
                dir_of_deskewed=dir_of_deskewed,
                dir_of_cropped_images=dir_of_cropped_images,
                dir_of_layout=dir_of_layout,
                image_filename_stem=Path(Path(image_filename).name).stem)
            self.writer = EynollahXmlWriter(
                dir_out=self.dir_out,
                image_filename=self.image_filename,
                curved_line=self.curved_line,
                textline_light = self.textline_light,
                pcgts=pcgts)
        self.logger = logger if logger else getLogger('eynollah')
        self.dir_models = dir_models
        self.model_dir_of_enhancement = dir_models + "/eynollah-enhancement_20210425"
        self.model_dir_of_binarization = dir_models + "/eynollah-binarization_20210425"
        self.model_dir_of_col_classifier = dir_models + "/eynollah-column-classifier_20210425"
        self.model_region_dir_p = dir_models + "/eynollah-main-regions-aug-scaling_20210425"
        self.model_region_dir_p2 = dir_models + "/eynollah-main-regions-aug-rotation_20210425"
        self.model_region_dir_fully_np = dir_models + "/modelens_full_lay_1__4_3_091124"#"/modelens_full_lay_1_3_031124"#"/modelens_full_lay_13__3_19_241024"#"/model_full_lay_13_241024"#"/modelens_full_lay_13_17_231024"#"/modelens_full_lay_1_2_221024"#"/eynollah-full-regions-1column_20210425"
        #self.model_region_dir_fully = dir_models + "/eynollah-full-regions-3+column_20210425"
        self.model_page_dir = dir_models + "/eynollah-page-extraction_20210425"
        self.model_region_dir_p_ens = dir_models + "/eynollah-main-regions-ensembled_20210425"
        self.model_region_dir_p_ens_light = dir_models + "/eynollah-main-regions_20220314"
        self.model_region_dir_p_ens_light_only_images_extraction = dir_models + "/eynollah-main-regions_20231127_672_org_ens_11_13_16_17_18"
        self.model_reading_order_machine_dir = dir_models + "/model_ens_reading_order_machine_based"
        self.model_region_dir_p_1_2_sp_np = dir_models + "/modelens_e_l_all_sp_0_1_2_3_4_171024"#"/modelens_12sp_elay_0_3_4__3_6_n"#"/modelens_earlylayout_12spaltige_2_3_5_6_7_8"#"/modelens_early12_sp_2_3_5_6_7_8_9_10_12_14_15_16_18"#"/modelens_1_2_4_5_early_lay_1_2_spaltige"#"/model_3_eraly_layout_no_patches_1_2_spaltige"
        ##self.model_region_dir_fully_new = dir_models + "/model_2_full_layout_new_trans"
        self.model_region_dir_fully = dir_models + "/modelens_full_lay_1__4_3_091124"#"/modelens_full_lay_1_3_031124"#"/modelens_full_lay_13__3_19_241024"#"/model_full_lay_13_241024"#"/modelens_full_lay_13_17_231024"#"/modelens_full_lay_1_2_221024"#"/modelens_full_layout_24_till_28"#"/model_2_full_layout_new_trans"
        if self.textline_light:
            self.model_textline_dir = dir_models + "/modelens_textline_0_1__2_4_16092024"#"/modelens_textline_1_4_16092024"#"/model_textline_ens_3_4_5_6_artificial"#"/modelens_textline_1_3_4_20240915"#"/model_textline_ens_3_4_5_6_artificial"#"/modelens_textline_9_12_13_14_15"#"/eynollah-textline_light_20210425"#
        else:
            self.model_textline_dir = dir_models + "/modelens_textline_0_1__2_4_16092024"#"/eynollah-textline_20210425"
        if self.ocr:
            self.model_ocr_dir = dir_models + "/checkpoint-166692_printed_trocr"
            
        self.model_tables = dir_models + "/eynollah-tables_20210319"
        
        self.models = {}
        
        if dir_in and light_version:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.compat.v1.Session(config=config)
            set_session(session)
            
            self.model_page = self.our_load_model(self.model_page_dir)
            self.model_classifier = self.our_load_model(self.model_dir_of_col_classifier)
            self.model_bin = self.our_load_model(self.model_dir_of_binarization)
            self.model_textline = self.our_load_model(self.model_textline_dir)
            self.model_region = self.our_load_model(self.model_region_dir_p_ens_light)
            self.model_region_1_2 = self.our_load_model(self.model_region_dir_p_1_2_sp_np)
            ###self.model_region_fl_new = self.our_load_model(self.model_region_dir_fully_new)
            self.model_region_fl_np = self.our_load_model(self.model_region_dir_fully_np)
            self.model_region_fl = self.our_load_model(self.model_region_dir_fully)
            self.model_reading_order_machine = self.our_load_model(self.model_reading_order_machine_dir)
            if self.ocr:
                self.model_ocr = VisionEncoderDecoderModel.from_pretrained(self.model_ocr_dir)
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")#("microsoft/trocr-base-printed")#("microsoft/trocr-base-handwritten")
            
            self.ls_imgs  = os.listdir(self.dir_in)
            
        if dir_in and self.extract_only_images:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.compat.v1.Session(config=config)
            set_session(session)

            self.model_page = self.our_load_model(self.model_page_dir)
            self.model_classifier = self.our_load_model(self.model_dir_of_col_classifier)
            self.model_bin = self.our_load_model(self.model_dir_of_binarization)
            #self.model_textline = self.our_load_model(self.model_textline_dir)
            self.model_region = self.our_load_model(self.model_region_dir_p_ens_light_only_images_extraction)
            #self.model_region_fl_np = self.our_load_model(self.model_region_dir_fully_np)
            #self.model_region_fl = self.our_load_model(self.model_region_dir_fully)

            self.ls_imgs  = os.listdir(self.dir_in)

        if dir_in and not (light_version or self.extract_only_images):
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.compat.v1.Session(config=config)
            set_session(session)
            
            self.model_page = self.our_load_model(self.model_page_dir)
            self.model_classifier = self.our_load_model(self.model_dir_of_col_classifier)
            self.model_bin = self.our_load_model(self.model_dir_of_binarization)
            self.model_textline = self.our_load_model(self.model_textline_dir)
            self.model_region = self.our_load_model(self.model_region_dir_p_ens)
            self.model_region_p2 = self.our_load_model(self.model_region_dir_p2)
            self.model_region_fl_np = self.our_load_model(self.model_region_dir_fully_np)
            self.model_region_fl = self.our_load_model(self.model_region_dir_fully)
            self.model_enhancement = self.our_load_model(self.model_dir_of_enhancement)
            self.model_reading_order_machine = self.our_load_model(self.model_reading_order_machine_dir)
            
            self.ls_imgs  = os.listdir(self.dir_in)
            
        
    def _cache_images(self, image_filename=None, image_pil=None):
        ret = {}
        t_c0 = time.time()
        if image_filename:
            ret['img'] = cv2.imread(image_filename)
            if self.light_version:
                self.dpi = 100
            else:
                self.dpi = check_dpi(image_filename)
        else:
            ret['img'] = pil2cv(image_pil)
            if self.light_version:
                self.dpi = 100
            else:
                self.dpi = check_dpi(image_pil)
        ret['img_grayscale'] = cv2.cvtColor(ret['img'], cv2.COLOR_BGR2GRAY)
        for prefix in ('',  '_grayscale'):
            ret[f'img{prefix}_uint8'] = ret[f'img{prefix}'].astype(np.uint8)
        return ret
    def reset_file_name_dir(self, image_filename):
        t_c = time.time()
        self._imgs = self._cache_images(image_filename=image_filename)
        self.image_filename = image_filename
        
        self.plotter = None if not self.enable_plotting else EynollahPlotter(
            dir_out=self.dir_out,
            dir_of_all=self.dir_of_all,
            dir_save_page=self.dir_save_page,
            dir_of_deskewed=self.dir_of_deskewed,
            dir_of_cropped_images=self.dir_of_cropped_images,
            dir_of_layout=self.dir_of_layout,
            image_filename_stem=Path(Path(image_filename).name).stem)
        
        self.writer = EynollahXmlWriter(
            dir_out=self.dir_out,
            image_filename=self.image_filename,
            curved_line=self.curved_line,
            textline_light = self.textline_light,
            pcgts=self.pcgts)
    def imread(self, grayscale=False, uint8=True):
        key = 'img'
        if grayscale:
            key += '_grayscale'
        if uint8:
            key += '_uint8'
        return self._imgs[key].copy()
    
    def isNaN(self, num):
        return num != num


    def predict_enhancement(self, img):
        self.logger.debug("enter predict_enhancement")
        model_enhancement, session_enhancement = self.start_new_session_and_model(self.model_dir_of_enhancement)

        img_height_model = model_enhancement.layers[len(model_enhancement.layers) - 1].output_shape[1]
        img_width_model = model_enhancement.layers[len(model_enhancement.layers) - 1].output_shape[2]
        if img.shape[0] < img_height_model:
            img = cv2.resize(img, (img.shape[1], img_width_model), interpolation=cv2.INTER_NEAREST)

        if img.shape[1] < img_width_model:
            img = cv2.resize(img, (img_height_model, img.shape[0]), interpolation=cv2.INTER_NEAREST)
        margin = int(0 * img_width_model)
        width_mid = img_width_model - 2 * margin
        height_mid = img_height_model - 2 * margin
        img = img / float(255.0)

        img_h = img.shape[0]
        img_w = img.shape[1]

        prediction_true = np.zeros((img_h, img_w, 3))
        nxf = img_w / float(width_mid)
        nyf = img_h / float(height_mid)

        nxf = int(nxf) + 1 if nxf > int(nxf) else int(nxf)
        nyf = int(nyf) + 1 if nyf > int(nyf) else int(nyf)

        for i in range(nxf):
            for j in range(nyf):
                if i == 0:
                    index_x_d = i * width_mid
                    index_x_u = index_x_d + img_width_model
                else:
                    index_x_d = i * width_mid
                    index_x_u = index_x_d + img_width_model
                if j == 0:
                    index_y_d = j * height_mid
                    index_y_u = index_y_d + img_height_model
                else:
                    index_y_d = j * height_mid
                    index_y_u = index_y_d + img_height_model

                if index_x_u > img_w:
                    index_x_u = img_w
                    index_x_d = img_w - img_width_model
                if index_y_u > img_h:
                    index_y_u = img_h
                    index_y_d = img_h - img_height_model

                img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
                label_p_pred = model_enhancement.predict(img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]),
                                                         verbose=0)

                seg = label_p_pred[0, :, :, :]
                seg = seg * 255

                if i == 0 and j == 0:
                    seg = seg[0 : seg.shape[0] - margin, 0 : seg.shape[1] - margin]
                    prediction_true[index_y_d + 0 : index_y_u - margin, index_x_d + 0 : index_x_u - margin, :] = seg
                elif i == nxf - 1 and j == nyf - 1:
                    seg = seg[margin : seg.shape[0] - 0, margin : seg.shape[1] - 0]
                    prediction_true[index_y_d + margin : index_y_u - 0, index_x_d + margin : index_x_u - 0, :] = seg
                elif i == 0 and j == nyf - 1:
                    seg = seg[margin : seg.shape[0] - 0, 0 : seg.shape[1] - margin]
                    prediction_true[index_y_d + margin : index_y_u - 0, index_x_d + 0 : index_x_u - margin, :] = seg
                elif i == nxf - 1 and j == 0:
                    seg = seg[0 : seg.shape[0] - margin, margin : seg.shape[1] - 0]
                    prediction_true[index_y_d + 0 : index_y_u - margin, index_x_d + margin : index_x_u - 0, :] = seg
                elif i == 0 and j != 0 and j != nyf - 1:
                    seg = seg[margin : seg.shape[0] - margin, 0 : seg.shape[1] - margin]
                    prediction_true[index_y_d + margin : index_y_u - margin, index_x_d + 0 : index_x_u - margin, :] = seg
                elif i == nxf - 1 and j != 0 and j != nyf - 1:
                    seg = seg[margin : seg.shape[0] - margin, margin : seg.shape[1] - 0]
                    prediction_true[index_y_d + margin : index_y_u - margin, index_x_d + margin : index_x_u - 0, :] = seg
                elif i != 0 and i != nxf - 1 and j == 0:
                    seg = seg[0 : seg.shape[0] - margin, margin : seg.shape[1] - margin]
                    prediction_true[index_y_d + 0 : index_y_u - margin, index_x_d + margin : index_x_u - margin, :] = seg
                elif i != 0 and i != nxf - 1 and j == nyf - 1:
                    seg = seg[margin : seg.shape[0] - 0, margin : seg.shape[1] - margin]
                    prediction_true[index_y_d + margin : index_y_u - 0, index_x_d + margin : index_x_u - margin, :] = seg
                else:
                    seg = seg[margin : seg.shape[0] - margin, margin : seg.shape[1] - margin]
                    prediction_true[index_y_d + margin : index_y_u - margin, index_x_d + margin : index_x_u - margin, :] = seg

        prediction_true = prediction_true.astype(int)
        return prediction_true

    def calculate_width_height_by_columns(self, img, num_col, width_early, label_p_pred):
        self.logger.debug("enter calculate_width_height_by_columns")
        if num_col == 1 and width_early < 1100:
            img_w_new = 2000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 2000)
        elif num_col == 1 and width_early >= 2500:
            img_w_new = 2000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 2000)
        elif num_col == 1 and width_early >= 1100 and width_early < 2500:
            img_w_new = width_early
            img_h_new = int(img.shape[0] / float(img.shape[1]) * width_early)
        elif num_col == 2 and width_early < 2000:
            img_w_new = 2400
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 2400)
        elif num_col == 2 and width_early >= 3500:
            img_w_new = 2400
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 2400)
        elif num_col == 2 and width_early >= 2000 and width_early < 3500:
            img_w_new = width_early
            img_h_new = int(img.shape[0] / float(img.shape[1]) * width_early)
        elif num_col == 3 and width_early < 2000:
            img_w_new = 3000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 3000)
        elif num_col == 3 and width_early >= 4000:
            img_w_new = 3000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 3000)
        elif num_col == 3 and width_early >= 2000 and width_early < 4000:
            img_w_new = width_early
            img_h_new = int(img.shape[0] / float(img.shape[1]) * width_early)
        elif num_col == 4 and width_early < 2500:
            img_w_new = 4000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 4000)
        elif num_col == 4 and width_early >= 5000:
            img_w_new = 4000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 4000)
        elif num_col == 4 and width_early >= 2500 and width_early < 5000:
            img_w_new = width_early
            img_h_new = int(img.shape[0] / float(img.shape[1]) * width_early)
        elif num_col == 5 and width_early < 3700:
            img_w_new = 5000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 5000)
        elif num_col == 5 and width_early >= 7000:
            img_w_new = 5000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 5000)
        elif num_col == 5 and width_early >= 3700 and width_early < 7000:
            img_w_new = width_early
            img_h_new = int(img.shape[0] / float(img.shape[1]) * width_early)
        elif num_col == 6 and width_early < 4500:
            img_w_new = 6500  # 5400
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 6500)
        else:
            img_w_new = width_early
            img_h_new = int(img.shape[0] / float(img.shape[1]) * width_early)

        if label_p_pred[0][int(num_col - 1)] < 0.9 and img_w_new < width_early:
            img_new = np.copy(img)
            num_column_is_classified = False
        #elif label_p_pred[0][int(num_col - 1)] < 0.8 and img_h_new >= 8000:
        elif img_h_new >= 8000:
            img_new = np.copy(img)
            num_column_is_classified = False
        else:
            img_new = resize_image(img, img_h_new, img_w_new)
            num_column_is_classified = True

        return img_new, num_column_is_classified
    
    def calculate_width_height_by_columns_1_2(self, img, num_col, width_early, label_p_pred):
        self.logger.debug("enter calculate_width_height_by_columns")
        if num_col == 1:
            img_w_new = 1000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 1000)
        else:
            img_w_new = 1300
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 1300)

        if label_p_pred[0][int(num_col - 1)] < 0.9 and img_w_new < width_early:
            img_new = np.copy(img)
            num_column_is_classified = False
        #elif label_p_pred[0][int(num_col - 1)] < 0.8 and img_h_new >= 8000:
        elif img_h_new >= 8000:
            img_new = np.copy(img)
            num_column_is_classified = False
        else:
            img_new = resize_image(img, img_h_new, img_w_new)
            num_column_is_classified = True

        return img_new, num_column_is_classified

    def calculate_width_height_by_columns_extract_only_images(self, img, num_col, width_early, label_p_pred):
        self.logger.debug("enter calculate_width_height_by_columns")
        if num_col == 1:
            img_w_new = 700
        elif num_col == 2:
            img_w_new = 900
        elif num_col == 3:
            img_w_new = 1500
        elif num_col == 4:
            img_w_new = 1800
        elif num_col == 5:
            img_w_new = 2200
        elif num_col == 6:
            img_w_new = 2500
        img_h_new = int(img.shape[0] / float(img.shape[1]) * img_w_new)

        img_new = resize_image(img, img_h_new, img_w_new)
        num_column_is_classified = True

        return img_new, num_column_is_classified

    def resize_image_with_column_classifier(self, is_image_enhanced, img_bin):
        self.logger.debug("enter resize_image_with_column_classifier")
        if self.input_binary:
            img = np.copy(img_bin)
        else:
            img = self.imread()

        _, page_coord = self.early_page_for_num_of_column_classification(img)
        
        if not self.dir_in:
            model_num_classifier, session_col_classifier = self.start_new_session_and_model(self.model_dir_of_col_classifier)
        if self.input_binary:
            img_in = np.copy(img)
            img_in = img_in / 255.0
            width_early = img_in.shape[1]
            img_in = cv2.resize(img_in, (448, 448), interpolation=cv2.INTER_NEAREST)
            img_in = img_in.reshape(1, 448, 448, 3)
        else:
            img_1ch = self.imread(grayscale=True, uint8=False)
            width_early = img_1ch.shape[1]
            img_1ch = img_1ch[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]

            # plt.imshow(img_1ch)
            # plt.show()
            img_1ch = img_1ch / 255.0

            img_1ch = cv2.resize(img_1ch, (448, 448), interpolation=cv2.INTER_NEAREST)

            img_in = np.zeros((1, img_1ch.shape[0], img_1ch.shape[1], 3))
            img_in[0, :, :, 0] = img_1ch[:, :]
            img_in[0, :, :, 1] = img_1ch[:, :]
            img_in[0, :, :, 2] = img_1ch[:, :]

        if not self.dir_in:
            label_p_pred = model_num_classifier.predict(img_in, verbose=0)
        else:
            label_p_pred = self.model_classifier.predict(img_in, verbose=0)

        num_col = np.argmax(label_p_pred[0]) + 1

        self.logger.info("Found %s columns (%s)", num_col, label_p_pred)

        img_new, _ = self.calculate_width_height_by_columns(img, num_col, width_early, label_p_pred)

        if img_new.shape[1] > img.shape[1]:
            img_new = self.predict_enhancement(img_new)
            is_image_enhanced = True

        return img, img_new, is_image_enhanced

    def resize_and_enhance_image_with_column_classifier(self,light_version):
        self.logger.debug("enter resize_and_enhance_image_with_column_classifier")
        dpi = self.dpi
        self.logger.info("Detected %s DPI", dpi)
        if self.input_binary:
            img = self.imread()
            if self.dir_in:
                prediction_bin = self.do_prediction(True, img, self.model_bin, n_batch_inference=5)
            else:
                
                model_bin, session_bin = self.start_new_session_and_model(self.model_dir_of_binarization)
                prediction_bin = self.do_prediction(True, img, model_bin, n_batch_inference=5)
            
            prediction_bin=prediction_bin[:,:,0]
            prediction_bin = (prediction_bin[:,:]==0)*1
            prediction_bin = prediction_bin*255
            
            prediction_bin =np.repeat(prediction_bin[:, :, np.newaxis], 3, axis=2)

            prediction_bin = prediction_bin.astype(np.uint8)
            img= np.copy(prediction_bin)
            img_bin = np.copy(prediction_bin)
        else:
            img = self.imread()
            img_bin = None
        
        width_early = img.shape[1]
        t1 = time.time()
        _, page_coord = self.early_page_for_num_of_column_classification(img_bin)
        
        self.image_page_org_size = img[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3], :]
        self.page_coord = page_coord
        
        if not self.dir_in:
            model_num_classifier, session_col_classifier = self.start_new_session_and_model(self.model_dir_of_col_classifier)
        
        if self.num_col_upper and not self.num_col_lower:
            num_col = self.num_col_upper
            label_p_pred = [np.ones(6)]
        elif self.num_col_lower and not self.num_col_upper:
            num_col = self.num_col_lower
            label_p_pred = [np.ones(6)]
        
        elif (not self.num_col_upper and not self.num_col_lower):
            if self.input_binary:
                img_in = np.copy(img)
                img_in = img_in / 255.0
                img_in = cv2.resize(img_in, (448, 448), interpolation=cv2.INTER_NEAREST)
                img_in = img_in.reshape(1, 448, 448, 3)
            else:
                img_1ch = self.imread(grayscale=True)
                width_early = img_1ch.shape[1]
                img_1ch = img_1ch[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]

                img_1ch = img_1ch / 255.0
                img_1ch = cv2.resize(img_1ch, (448, 448), interpolation=cv2.INTER_NEAREST)
                img_in = np.zeros((1, img_1ch.shape[0], img_1ch.shape[1], 3))
                img_in[0, :, :, 0] = img_1ch[:, :]
                img_in[0, :, :, 1] = img_1ch[:, :]
                img_in[0, :, :, 2] = img_1ch[:, :]


            if self.dir_in:
                label_p_pred = self.model_classifier.predict(img_in, verbose=0)
            else:
                label_p_pred = model_num_classifier.predict(img_in, verbose=0)
            num_col = np.argmax(label_p_pred[0]) + 1
        elif (self.num_col_upper and self.num_col_lower) and (self.num_col_upper!=self.num_col_lower):
            if self.input_binary:
                img_in = np.copy(img)
                img_in = img_in / 255.0
                img_in = cv2.resize(img_in, (448, 448), interpolation=cv2.INTER_NEAREST)
                img_in = img_in.reshape(1, 448, 448, 3)
            else:
                img_1ch = self.imread(grayscale=True)
                width_early = img_1ch.shape[1]
                img_1ch = img_1ch[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]

                img_1ch = img_1ch / 255.0
                img_1ch = cv2.resize(img_1ch, (448, 448), interpolation=cv2.INTER_NEAREST)
                img_in = np.zeros((1, img_1ch.shape[0], img_1ch.shape[1], 3))
                img_in[0, :, :, 0] = img_1ch[:, :]
                img_in[0, :, :, 1] = img_1ch[:, :]
                img_in[0, :, :, 2] = img_1ch[:, :]


            if self.dir_in:
                label_p_pred = self.model_classifier.predict(img_in, verbose=0)
            else:
                label_p_pred = model_num_classifier.predict(img_in, verbose=0)
            num_col = np.argmax(label_p_pred[0]) + 1
            
            if num_col > self.num_col_upper:
                num_col = self.num_col_upper
                label_p_pred = [np.ones(6)]
            if num_col < self.num_col_lower:
                num_col = self.num_col_lower
                label_p_pred = [np.ones(6)]
                
        else:
            num_col = self.num_col_upper
            label_p_pred = [np.ones(6)]
                
        
        self.logger.info("Found %d columns (%s)", num_col, np.around(label_p_pred, decimals=5))

        if not self.extract_only_images:
            if dpi < DPI_THRESHOLD:
                if light_version and num_col in (1,2):
                    img_new, num_column_is_classified = self.calculate_width_height_by_columns_1_2(img, num_col, width_early, label_p_pred)
                else:
                    img_new, num_column_is_classified = self.calculate_width_height_by_columns(img, num_col, width_early, label_p_pred)
                if light_version:
                    image_res = np.copy(img_new)
                else:
                    image_res = self.predict_enhancement(img_new)
                is_image_enhanced = True
            else:
                if light_version and num_col in (1,2):
                    img_new, num_column_is_classified = self.calculate_width_height_by_columns_1_2(img, num_col, width_early, label_p_pred)
                    image_res = np.copy(img_new)
                    is_image_enhanced = True
                else:
                    num_column_is_classified = True
                    image_res = np.copy(img)
                    is_image_enhanced = False
        else:
            num_column_is_classified = True
            image_res = np.copy(img)
            is_image_enhanced = False

        self.logger.debug("exit resize_and_enhance_image_with_column_classifier")
        return is_image_enhanced, img, image_res, num_col, num_column_is_classified, img_bin

    # pylint: disable=attribute-defined-outside-init
    def get_image_and_scales(self, img_org, img_res, scale):
        self.logger.debug("enter get_image_and_scales")
        self.image = np.copy(img_res)
        self.image_org = np.copy(img_org)
        self.height_org = self.image.shape[0]
        self.width_org = self.image.shape[1]

        self.img_hight_int = int(self.image.shape[0] * scale)
        self.img_width_int = int(self.image.shape[1] * scale)
        self.scale_y = self.img_hight_int / float(self.image.shape[0])
        self.scale_x = self.img_width_int / float(self.image.shape[1])

        self.image = resize_image(self.image, self.img_hight_int, self.img_width_int)

        # Also set for the plotter
        if self.plotter:
            self.plotter.image_org = self.image_org
            self.plotter.scale_y = self.scale_y
            self.plotter.scale_x = self.scale_x
        # Also set for the writer
        self.writer.image_org = self.image_org
        self.writer.scale_y = self.scale_y
        self.writer.scale_x = self.scale_x
        self.writer.height_org = self.height_org
        self.writer.width_org = self.width_org

    def get_image_and_scales_after_enhancing(self, img_org, img_res):
        self.logger.debug("enter get_image_and_scales_after_enhancing")
        self.image = np.copy(img_res)
        self.image = self.image.astype(np.uint8)
        self.image_org = np.copy(img_org)
        self.height_org = self.image_org.shape[0]
        self.width_org = self.image_org.shape[1]

        self.scale_y = img_res.shape[0] / float(self.image_org.shape[0])
        self.scale_x = img_res.shape[1] / float(self.image_org.shape[1])

        # Also set for the plotter
        if self.plotter:
            self.plotter.image_org = self.image_org
            self.plotter.scale_y = self.scale_y
            self.plotter.scale_x = self.scale_x
        # Also set for the writer
        self.writer.image_org = self.image_org
        self.writer.scale_y = self.scale_y
        self.writer.scale_x = self.scale_x
        self.writer.height_org = self.height_org
        self.writer.width_org = self.width_org

    def start_new_session_and_model_old(self, model_dir):
        self.logger.debug("enter start_new_session_and_model (model_dir=%s)", model_dir)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        session = tf.InteractiveSession()
        model = load_model(model_dir, compile=False)

        return model, session

    
    def start_new_session_and_model(self, model_dir):
        self.logger.debug("enter start_new_session_and_model (model_dir=%s)", model_dir)
        #gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        #gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=7.7, allow_growth=True)
        #session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except:
            self.logger.warning("no GPU device available")

        if model_dir.endswith('.h5') and Path(model_dir[:-3]).exists():
            # prefer SavedModel over HDF5 format if it exists
            model_dir = model_dir[:-3]
        if model_dir in self.models:
            model = self.models[model_dir]
        else:
            try:
                model = load_model(model_dir, compile=False)
                self.models[model_dir] = model
            except:
                model = load_model(model_dir , compile=False,custom_objects = {"PatchEncoder": PatchEncoder, "Patches": Patches})
                self.models[model_dir] = model


        return model, None

    def do_prediction(self, patches, img, model, n_batch_inference=1, marginal_of_patch_percent=0.1, thresholding_for_some_classes_in_light_version=False, thresholding_for_artificial_class_in_light_version=False):
        self.logger.debug("enter do_prediction")

        img_height_model = model.layers[len(model.layers) - 1].output_shape[1]
        img_width_model = model.layers[len(model.layers) - 1].output_shape[2]

        if not patches:
            img_h_page = img.shape[0]
            img_w_page = img.shape[1]
            img = img / float(255.0)
            img = resize_image(img, img_height_model, img_width_model)

            label_p_pred = model.predict(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]),
                                         verbose=0)

            seg = np.argmax(label_p_pred, axis=3)[0]
            
            if thresholding_for_artificial_class_in_light_version:
                seg_art = label_p_pred[0,:,:,2]
                
                seg_art[seg_art<0.2] = 0
                seg_art[seg_art>0] =1
                
                seg[seg_art==1]=2
            seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
            prediction_true = resize_image(seg_color, img_h_page, img_w_page)
            prediction_true = prediction_true.astype(np.uint8)


        else:
            if img.shape[0] < img_height_model:
                img = resize_image(img, img_height_model, img.shape[1])

            if img.shape[1] < img_width_model:
                img = resize_image(img, img.shape[0], img_width_model)

            self.logger.debug("Patch size: %sx%s", img_height_model, img_width_model)
            margin = int(marginal_of_patch_percent * img_height_model)
            width_mid = img_width_model - 2 * margin
            height_mid = img_height_model - 2 * margin
            img = img / float(255.0)
            #img = img.astype(np.float16)
            img_h = img.shape[0]
            img_w = img.shape[1]
            prediction_true = np.zeros((img_h, img_w, 3))
            mask_true = np.zeros((img_h, img_w))
            nxf = img_w / float(width_mid)
            nyf = img_h / float(height_mid)
            nxf = int(nxf) + 1 if nxf > int(nxf) else int(nxf)
            nyf = int(nyf) + 1 if nyf > int(nyf) else int(nyf)
            
            list_i_s = []
            list_j_s = []
            list_x_u = []
            list_x_d = []
            list_y_u = []
            list_y_d = []
            
            batch_indexer = 0
            
            img_patch = np.zeros((n_batch_inference, img_height_model, img_width_model, 3))
            for i in range(nxf):
                for j in range(nyf):
                    if i == 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model
                    else:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model
                    if j == 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model
                    else:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model
                    if index_x_u > img_w:
                        index_x_u = img_w
                        index_x_d = img_w - img_width_model
                    if index_y_u > img_h:
                        index_y_u = img_h
                        index_y_d = img_h - img_height_model
                    
                    list_i_s.append(i)
                    list_j_s.append(j)
                    list_x_u.append(index_x_u)
                    list_x_d.append(index_x_d)
                    list_y_d.append(index_y_d)
                    list_y_u.append(index_y_u)
                    

                    img_patch[batch_indexer,:,:,:] = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
                    
                    batch_indexer = batch_indexer + 1
                    
                    if batch_indexer == n_batch_inference:
                        label_p_pred = model.predict(img_patch,verbose=0)
                        
                        seg = np.argmax(label_p_pred, axis=3)
                        
                        if thresholding_for_some_classes_in_light_version:
                            seg_not_base = label_p_pred[:,:,:,4]
                            seg_not_base[seg_not_base>0.03] =1
                            seg_not_base[seg_not_base<1] =0
                            
                            seg_line = label_p_pred[:,:,:,3]
                            seg_line[seg_line>0.1] =1
                            seg_line[seg_line<1] =0
                            
                            seg_background = label_p_pred[:,:,:,0]
                            seg_background[seg_background>0.25] =1
                            seg_background[seg_background<1] =0
                            
                            seg[seg_not_base==1]=4
                            seg[seg_background==1]=0
                            seg[(seg_line==1) & (seg==0)]=3
                        if thresholding_for_artificial_class_in_light_version:
                            seg_art = label_p_pred[:,:,:,2]
                            
                            seg_art[seg_art<0.2] = 0
                            seg_art[seg_art>0] =1
                            
                            seg[seg_art==1]=2
                        
                        indexer_inside_batch = 0
                        for i_batch, j_batch in zip(list_i_s, list_j_s):
                            seg_in = seg[indexer_inside_batch,:,:]
                            seg_color = np.repeat(seg_in[:, :, np.newaxis], 3, axis=2)
                            
                            index_y_u_in = list_y_u[indexer_inside_batch]
                            index_y_d_in = list_y_d[indexer_inside_batch]
                            
                            index_x_u_in = list_x_u[indexer_inside_batch]
                            index_x_d_in = list_x_d[indexer_inside_batch]
                            
                            if i_batch == 0 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch == 0 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch == 0 and j_batch != 0 and j_batch != nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch != 0 and j_batch != nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch != 0 and i_batch != nxf - 1 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                            elif i_batch != 0 and i_batch != nxf - 1 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                            else:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                                
                            indexer_inside_batch = indexer_inside_batch +1
                                
                        
                        list_i_s = []
                        list_j_s = []
                        list_x_u = []
                        list_x_d = []
                        list_y_u = []
                        list_y_d = []
                        
                        batch_indexer = 0
                        
                        img_patch = np.zeros((n_batch_inference, img_height_model, img_width_model, 3))
                        
                    elif i==(nxf-1) and j==(nyf-1):
                        label_p_pred = model.predict(img_patch,verbose=0)
                        
                        seg = np.argmax(label_p_pred, axis=3)
                        if thresholding_for_some_classes_in_light_version:
                            seg_not_base = label_p_pred[:,:,:,4]
                            seg_not_base[seg_not_base>0.03] =1
                            seg_not_base[seg_not_base<1] =0
                            
                            seg_line = label_p_pred[:,:,:,3]
                            seg_line[seg_line>0.1] =1
                            seg_line[seg_line<1] =0
                            
                            seg_background = label_p_pred[:,:,:,0]
                            seg_background[seg_background>0.25] =1
                            seg_background[seg_background<1] =0
                            
                            seg[seg_not_base==1]=4
                            seg[seg_background==1]=0
                            seg[(seg_line==1) & (seg==0)]=3
                            
                        if thresholding_for_artificial_class_in_light_version:
                            seg_art = label_p_pred[:,:,:,2]
                            
                            seg_art[seg_art<0.2] = 0
                            seg_art[seg_art>0] =1
                            
                            seg[seg_art==1]=2
                        
                        indexer_inside_batch = 0
                        for i_batch, j_batch in zip(list_i_s, list_j_s):
                            seg_in = seg[indexer_inside_batch,:,:]
                            seg_color = np.repeat(seg_in[:, :, np.newaxis], 3, axis=2)
                            
                            index_y_u_in = list_y_u[indexer_inside_batch]
                            index_y_d_in = list_y_d[indexer_inside_batch]
                            
                            index_x_u_in = list_x_u[indexer_inside_batch]
                            index_x_d_in = list_x_d[indexer_inside_batch]
                            
                            if i_batch == 0 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch == 0 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch == 0 and j_batch != 0 and j_batch != nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch != 0 and j_batch != nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch != 0 and i_batch != nxf - 1 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                            elif i_batch != 0 and i_batch != nxf - 1 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                            else:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                                
                            indexer_inside_batch = indexer_inside_batch +1
                                
                        
                        list_i_s = []
                        list_j_s = []
                        list_x_u = []
                        list_x_d = []
                        list_y_u = []
                        list_y_d = []
                        
                        batch_indexer = 0
                        
                        img_patch = np.zeros((n_batch_inference, img_height_model, img_width_model, 3))
                        
            prediction_true = prediction_true.astype(np.uint8)
        #del model
        #gc.collect()
        return prediction_true
    def do_padding_with_scale(self,img, scale):
        h_n = int(img.shape[0]*scale)
        w_n = int(img.shape[1]*scale)
        
        channel0_avg = int( np.mean(img[:,:,0]) )
        channel1_avg = int( np.mean(img[:,:,1]) )
        channel2_avg = int( np.mean(img[:,:,2]) )
        
        h_diff = img.shape[0] - h_n
        w_diff = img.shape[1] - w_n
        
        h_start = int(h_diff / 2.)
        w_start = int(w_diff / 2.)
        
        img_res = resize_image(img, h_n, w_n)
        #label_res = resize_image(label, h_n, w_n)
        
        img_scaled_padded = np.copy(img)
        
        #label_scaled_padded = np.zeros(label.shape)
        
        img_scaled_padded[:,:,0] = channel0_avg
        img_scaled_padded[:,:,1] = channel1_avg
        img_scaled_padded[:,:,2] = channel2_avg
        
        img_scaled_padded[h_start:h_start+h_n, w_start:w_start+w_n,:] = img_res[:,:,:]
        #label_scaled_padded[h_start:h_start+h_n, w_start:w_start+w_n,:] = label_res[:,:,:]
        
        return img_scaled_padded#, label_scaled_padded
    def do_prediction_new_concept(self, patches, img, model, n_batch_inference=1, marginal_of_patch_percent=0.1, thresholding_for_some_classes_in_light_version=False, thresholding_for_artificial_class_in_light_version=False):
        self.logger.debug("enter do_prediction")

        img_height_model = model.layers[len(model.layers) - 1].output_shape[1]
        img_width_model = model.layers[len(model.layers) - 1].output_shape[2]

        if not patches:
            img_h_page = img.shape[0]
            img_w_page = img.shape[1]
            img = img / 255.0
            img = resize_image(img, img_height_model, img_width_model)

            label_p_pred = model.predict(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]), verbose=0)
            seg = np.argmax(label_p_pred, axis=3)[0]
            
            if thresholding_for_artificial_class_in_light_version:
                #seg_text = label_p_pred[0,:,:,1]
                #seg_text[seg_text<0.2] =0
                #seg_text[seg_text>0] =1
                #seg[seg_text==1]=1
                
                seg_art = label_p_pred[0,:,:,4]
                seg_art[seg_art<0.2] =0
                seg_art[seg_art>0] =1
                seg[seg_art==1]=4

            
            seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
            prediction_true = resize_image(seg_color, img_h_page, img_w_page)
            prediction_true = prediction_true.astype(np.uint8)


        else:
            if img.shape[0] < img_height_model:
                img = resize_image(img, img_height_model, img.shape[1])

            if img.shape[1] < img_width_model:
                img = resize_image(img, img.shape[0], img_width_model)

            self.logger.debug("Patch size: %sx%s", img_height_model, img_width_model)
            margin = int(marginal_of_patch_percent * img_height_model)
            width_mid = img_width_model - 2 * margin
            height_mid = img_height_model - 2 * margin
            img = img / float(255.0)
            img = img.astype(np.float16)
            img_h = img.shape[0]
            img_w = img.shape[1]
            prediction_true = np.zeros((img_h, img_w, 3))
            mask_true = np.zeros((img_h, img_w))
            nxf = img_w / float(width_mid)
            nyf = img_h / float(height_mid)
            nxf = int(nxf) + 1 if nxf > int(nxf) else int(nxf)
            nyf = int(nyf) + 1 if nyf > int(nyf) else int(nyf)
            
            list_i_s = []
            list_j_s = []
            list_x_u = []
            list_x_d = []
            list_y_u = []
            list_y_d = []
            
            batch_indexer = 0
            img_patch = np.zeros((n_batch_inference, img_height_model, img_width_model, 3))

            for i in range(nxf):
                for j in range(nyf):
                    if i == 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model
                    else:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model
                    if j == 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model
                    else:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model
                    if index_x_u > img_w:
                        index_x_u = img_w
                        index_x_d = img_w - img_width_model
                    if index_y_u > img_h:
                        index_y_u = img_h
                        index_y_d = img_h - img_height_model
                        
                        
                    list_i_s.append(i)
                    list_j_s.append(j)
                    list_x_u.append(index_x_u)
                    list_x_d.append(index_x_d)
                    list_y_d.append(index_y_d)
                    list_y_u.append(index_y_u)
                    

                    img_patch[batch_indexer,:,:,:] = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
                    
                    batch_indexer = batch_indexer + 1

                    if batch_indexer == n_batch_inference:
                        label_p_pred = model.predict(img_patch,verbose=0)
                        
                        seg = np.argmax(label_p_pred, axis=3)
                        
                        if thresholding_for_some_classes_in_light_version:
                            seg_art = label_p_pred[:,:,:,4]
                            seg_art[seg_art<0.2] =0
                            seg_art[seg_art>0] =1
                            
                            seg_line = label_p_pred[:,:,:,3]
                            seg_line[seg_line>0.1] =1
                            seg_line[seg_line<1] =0
                            
                            seg[seg_art==1]=4
                            seg[(seg_line==1) & (seg==0)]=3
                        if thresholding_for_artificial_class_in_light_version:
                            seg_art = label_p_pred[:,:,:,2]
                            
                            seg_art[seg_art<0.2] = 0
                            seg_art[seg_art>0] =1
                            
                            seg[seg_art==1]=2
                        
                        indexer_inside_batch = 0
                        for i_batch, j_batch in zip(list_i_s, list_j_s):
                            seg_in = seg[indexer_inside_batch,:,:]
                            seg_color = np.repeat(seg_in[:, :, np.newaxis], 3, axis=2)
                            
                            index_y_u_in = list_y_u[indexer_inside_batch]
                            index_y_d_in = list_y_d[indexer_inside_batch]
                            
                            index_x_u_in = list_x_u[indexer_inside_batch]
                            index_x_d_in = list_x_d[indexer_inside_batch]
                            
                            if i_batch == 0 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch == 0 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch == 0 and j_batch != 0 and j_batch != nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch != 0 and j_batch != nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch != 0 and i_batch != nxf - 1 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                            elif i_batch != 0 and i_batch != nxf - 1 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                            else:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                                
                            indexer_inside_batch = indexer_inside_batch +1
                                
                        
                        list_i_s = []
                        list_j_s = []
                        list_x_u = []
                        list_x_d = []
                        list_y_u = []
                        list_y_d = []
                        
                        batch_indexer = 0
                        
                        img_patch = np.zeros((n_batch_inference, img_height_model, img_width_model, 3))
                        
                    elif i==(nxf-1) and j==(nyf-1):
                        label_p_pred = model.predict(img_patch,verbose=0)
                        
                        seg = np.argmax(label_p_pred, axis=3)
                        if thresholding_for_some_classes_in_light_version:
                            seg_art = label_p_pred[:,:,:,4]
                            seg_art[seg_art<0.2] =0
                            seg_art[seg_art>0] =1
                            
                            seg_line = label_p_pred[:,:,:,3]
                            seg_line[seg_line>0.1] =1
                            seg_line[seg_line<1] =0
                            
                            seg[seg_art==1]=4
                            seg[(seg_line==1) & (seg==0)]=3
                            
                        if thresholding_for_artificial_class_in_light_version:
                            seg_art = label_p_pred[:,:,:,2]
                            
                            seg_art[seg_art<0.2] = 0
                            seg_art[seg_art>0] =1
                            
                            seg[seg_art==1]=2
                        
                        indexer_inside_batch = 0
                        for i_batch, j_batch in zip(list_i_s, list_j_s):
                            seg_in = seg[indexer_inside_batch,:,:]
                            seg_color = np.repeat(seg_in[:, :, np.newaxis], 3, axis=2)
                            
                            index_y_u_in = list_y_u[indexer_inside_batch]
                            index_y_d_in = list_y_d[indexer_inside_batch]
                            
                            index_x_u_in = list_x_u[indexer_inside_batch]
                            index_x_d_in = list_x_d[indexer_inside_batch]
                            
                            if i_batch == 0 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch == 0 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch == 0 and j_batch != 0 and j_batch != nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch != 0 and j_batch != nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch != 0 and i_batch != nxf - 1 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                            elif i_batch != 0 and i_batch != nxf - 1 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                            else:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                                
                            indexer_inside_batch = indexer_inside_batch +1
                        
                        list_i_s = []
                        list_j_s = []
                        list_x_u = []
                        list_x_d = []
                        list_y_u = []
                        list_y_d = []
                        
                        batch_indexer = 0
                        img_patch = np.zeros((n_batch_inference, img_height_model, img_width_model, 3))

            prediction_true = prediction_true.astype(np.uint8)
        return prediction_true

    def extract_page(self):
        self.logger.debug("enter extract_page")
        cont_page = []
        if not self.ignore_page_extraction:
            img = cv2.GaussianBlur(self.image, (5, 5), 0)
            
            if not self.dir_in:
                model_page, session_page = self.start_new_session_and_model(self.model_page_dir)
                
            if not self.dir_in:
                img_page_prediction = self.do_prediction(False, img, model_page)
            else:
                img_page_prediction = self.do_prediction(False, img, self.model_page)
            imgray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(imgray, 0, 255, 0)
            thresh = cv2.dilate(thresh, KERNEL, iterations=3)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours)>0:
                cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])
                cnt = contours[np.argmax(cnt_size)]
                x, y, w, h = cv2.boundingRect(cnt)
                if x <= 30:
                    w += x
                    x = 0
                if (self.image.shape[1] - (x + w)) <= 30:
                    w = w + (self.image.shape[1] - (x + w))
                if y <= 30:
                    h = h + y
                    y = 0
                if (self.image.shape[0] - (y + h)) <= 30:
                    h = h + (self.image.shape[0] - (y + h))

                box = [x, y, w, h]
            else:
                box = [0, 0, img.shape[1], img.shape[0]]
            croped_page, page_coord = crop_image_inside_box(box, self.image)
            cont_page.append(np.array([[page_coord[2], page_coord[0]], [page_coord[3], page_coord[0]], [page_coord[3], page_coord[1]], [page_coord[2], page_coord[1]]]))
            
            self.logger.debug("exit extract_page")
        else:
            box = [0, 0, self.image.shape[1], self.image.shape[0]]
            croped_page, page_coord = crop_image_inside_box(box, self.image)
            cont_page.append(np.array([[page_coord[2], page_coord[0]], [page_coord[3], page_coord[0]], [page_coord[3], page_coord[1]], [page_coord[2], page_coord[1]]]))
        return croped_page, page_coord, cont_page

    def early_page_for_num_of_column_classification(self,img_bin):
        if not self.ignore_page_extraction:
            self.logger.debug("enter early_page_for_num_of_column_classification")
            if self.input_binary:
                img =np.copy(img_bin)
                img = img.astype(np.uint8)
            else:
                img = self.imread()
            if not self.dir_in:
                model_page, session_page = self.start_new_session_and_model(self.model_page_dir)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
            if self.dir_in:
                img_page_prediction = self.do_prediction(False, img, self.model_page)
            else:
                img_page_prediction = self.do_prediction(False, img, model_page)

            imgray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(imgray, 0, 255, 0)
            thresh = cv2.dilate(thresh, KERNEL, iterations=3)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours)>0:
                cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])
                cnt = contours[np.argmax(cnt_size)]
                x, y, w, h = cv2.boundingRect(cnt)
                box = [x, y, w, h]
            else:
                box = [0, 0, img.shape[1], img.shape[0]]
            croped_page, page_coord = crop_image_inside_box(box, img)
            
            self.logger.debug("exit early_page_for_num_of_column_classification")
        else:
            img = self.imread()
            box = [0, 0, img.shape[1], img.shape[0]]
            croped_page, page_coord = crop_image_inside_box(box, img)
        return croped_page, page_coord

    def extract_text_regions_new(self, img, patches, cols):
        self.logger.debug("enter extract_text_regions")
        img_height_h = img.shape[0]
        img_width_h = img.shape[1]
        if not self.dir_in:
            model_region, session_region = self.start_new_session_and_model(self.model_region_dir_fully if patches else self.model_region_dir_fully_np)
        else:
            model_region = self.model_region_fl if patches else self.model_region_fl_np

        if not patches:
            if self.light_version:
                pass
            else:
                img = otsu_copy_binary(img)
            #img = img.astype(np.uint8)
            prediction_regions2 = None
        else:
            if cols == 1:
                if self.light_version:
                    pass
                else:
                    img = otsu_copy_binary(img)
                img = img.astype(np.uint8)

                img = resize_image(img, int(img_height_h * 1000 / float(img_width_h)), 1000)
                img = img.astype(np.uint8)

            if cols == 2:
                if self.light_version:
                    pass
                else:
                    img = otsu_copy_binary(img)
                img = img.astype(np.uint8)
                img = resize_image(img, int(img_height_h * 1300 / float(img_width_h)), 1300)
                img = img.astype(np.uint8)

            if cols == 3:
                if self.light_version:
                    pass
                else:
                    img = otsu_copy_binary(img)
                img = img.astype(np.uint8)
                img = resize_image(img, int(img_height_h * 1600 / float(img_width_h)), 1600)
                img = img.astype(np.uint8)

            if cols == 4:
                if self.light_version:
                    pass
                else:
                    img = otsu_copy_binary(img)
                img = img.astype(np.uint8)
                img = resize_image(img, int(img_height_h * 1900 / float(img_width_h)), 1900)
                img = img.astype(np.uint8)
                
            if cols == 5:
                if self.light_version:
                    pass
                else:
                    img = otsu_copy_binary(img)
                img = img.astype(np.uint8)
                img = resize_image(img, int(img_height_h * 2200 / float(img_width_h)), 2200)
                img = img.astype(np.uint8)

            if cols >= 6:
                if self.light_version:
                    pass
                else:
                    img = otsu_copy_binary(img)
                img = img.astype(np.uint8)
                img = resize_image(img, int(img_height_h * 2500 / float(img_width_h)), 2500)
                img = img.astype(np.uint8)

        marginal_of_patch_percent = 0.1
        
        prediction_regions = self.do_prediction(patches, img, model_region, marginal_of_patch_percent=marginal_of_patch_percent, n_batch_inference=3)
        
        
        ##prediction_regions = self.do_prediction(False, img, model_region, marginal_of_patch_percent=marginal_of_patch_percent, n_batch_inference=3)
        
        prediction_regions = resize_image(prediction_regions, img_height_h, img_width_h)
        self.logger.debug("exit extract_text_regions")
        return prediction_regions, prediction_regions
    
    
    def extract_text_regions(self, img, patches, cols):
        self.logger.debug("enter extract_text_regions")
        img_height_h = img.shape[0]
        img_width_h = img.shape[1]
        if not self.dir_in:
            model_region, session_region = self.start_new_session_and_model(self.model_region_dir_fully if patches else self.model_region_dir_fully_np)
        else:
            model_region = self.model_region_fl if patches else self.model_region_fl_np

        if not patches:
            img = otsu_copy_binary(img)
            img = img.astype(np.uint8)
            prediction_regions2 = None
        else:
            if cols == 1:
                img2 = otsu_copy_binary(img)
                img2 = img2.astype(np.uint8)
                img2 = resize_image(img2, int(img_height_h * 0.7), int(img_width_h * 0.7))
                marginal_of_patch_percent = 0.1
                prediction_regions2 = self.do_prediction(patches, img2, model_region, marginal_of_patch_percent=marginal_of_patch_percent)
                prediction_regions2 = resize_image(prediction_regions2, img_height_h, img_width_h)

            if cols == 2:
                img2 = otsu_copy_binary(img)
                img2 = img2.astype(np.uint8)
                img2 = resize_image(img2, int(img_height_h * 0.4), int(img_width_h * 0.4))
                marginal_of_patch_percent = 0.1
                prediction_regions2 = self.do_prediction(patches, img2, model_region, marginal_of_patch_percent=marginal_of_patch_percent)
                prediction_regions2 = resize_image(prediction_regions2, img_height_h, img_width_h)

            elif cols > 2:
                img2 = otsu_copy_binary(img)
                img2 = img2.astype(np.uint8)
                img2 = resize_image(img2, int(img_height_h * 0.3), int(img_width_h * 0.3))
                marginal_of_patch_percent = 0.1
                prediction_regions2 = self.do_prediction(patches, img2, model_region, marginal_of_patch_percent=marginal_of_patch_percent)
                prediction_regions2 = resize_image(prediction_regions2, img_height_h, img_width_h)

            if cols == 2:
                img = otsu_copy_binary(img)
                img = img.astype(np.uint8)
                if img_width_h >= 2000:
                    img = resize_image(img, int(img_height_h * 0.9), int(img_width_h * 0.9))
                img = img.astype(np.uint8)

            if cols == 1:
                img = otsu_copy_binary(img)
                img = img.astype(np.uint8)
                img = resize_image(img, int(img_height_h * 0.5), int(img_width_h * 0.5))
                img = img.astype(np.uint8)

            if cols == 3:
                if (self.scale_x == 1 and img_width_h > 3000) or (self.scale_x != 1 and img_width_h > 2800):
                    img = otsu_copy_binary(img)
                    img = img.astype(np.uint8)
                    img = resize_image(img, int(img_height_h * 2800 / float(img_width_h)), 2800)
                else:
                    img = otsu_copy_binary(img)
                    img = img.astype(np.uint8)

            if cols == 4:
                if (self.scale_x == 1 and img_width_h > 4000) or (self.scale_x != 1 and img_width_h > 3700):
                    img = otsu_copy_binary(img)
                    img = img.astype(np.uint8)
                    img= resize_image(img, int(img_height_h * 3700 / float(img_width_h)), 3700)
                else:
                    img = otsu_copy_binary(img)
                    img = img.astype(np.uint8)
                    img= resize_image(img, int(img_height_h * 0.9), int(img_width_h * 0.9))

            if cols == 5:
                if self.scale_x == 1 and img_width_h > 5000:
                    img = otsu_copy_binary(img)
                    img = img.astype(np.uint8)
                    img= resize_image(img, int(img_height_h * 0.7), int(img_width_h * 0.7))
                else:
                    img = otsu_copy_binary(img)
                    img = img.astype(np.uint8)
                    img= resize_image(img, int(img_height_h * 0.9), int(img_width_h * 0.9) )

            if cols >= 6:
                if img_width_h > 5600:
                    img = otsu_copy_binary(img)
                    img = img.astype(np.uint8)
                    img= resize_image(img, int(img_height_h * 5600 / float(img_width_h)), 5600)
                else:
                    img = otsu_copy_binary(img)
                    img = img.astype(np.uint8)
                    img= resize_image(img, int(img_height_h * 0.9), int(img_width_h * 0.9))

        marginal_of_patch_percent = 0.1
        prediction_regions = self.do_prediction(patches, img, model_region, marginal_of_patch_percent=marginal_of_patch_percent)
        prediction_regions = resize_image(prediction_regions, img_height_h, img_width_h)
        self.logger.debug("exit extract_text_regions")
        return prediction_regions, prediction_regions2
    
    def get_slopes_and_deskew_new_light2(self, contours, contours_par, textline_mask_tot, image_page_rotated, boxes, slope_deskew):
        
        polygons_of_textlines = return_contours_of_interested_region(textline_mask_tot,1,0.00001)
        
        
        M_main_tot = [cv2.moments(polygons_of_textlines[j]) for j in range(len(polygons_of_textlines))]
        cx_main_tot = [(M_main_tot[j]["m10"] / (M_main_tot[j]["m00"] + 1e-32)) for j in range(len(M_main_tot))]
        cy_main_tot = [(M_main_tot[j]["m01"] / (M_main_tot[j]["m00"] + 1e-32)) for j in range(len(M_main_tot))]
        
        args_textlines = np.array(range(len(polygons_of_textlines)))
        all_found_textline_polygons = []
        slopes = []
        all_box_coord =[]
        
        for index, con_region_ind in enumerate(contours_par):
            results = [cv2.pointPolygonTest(con_region_ind, (cx_main_tot[ind], cy_main_tot[ind]), False) for ind in args_textlines ]
            results = np.array(results)
            
            indexes_in = args_textlines[results==1]
            
            textlines_ins = [polygons_of_textlines[ind] for ind in indexes_in]
            
            all_found_textline_polygons.append(textlines_ins)
            slopes.append(0)
            
            _, crop_coor = crop_image_inside_box(boxes[index],image_page_rotated)
            
            all_box_coord.append(crop_coor)
            
        return slopes, all_found_textline_polygons, boxes, contours, contours_par, all_box_coord, np.array(range(len(contours_par)))
        
    def get_slopes_and_deskew_new_light(self, contours, contours_par, textline_mask_tot, image_page_rotated, boxes, slope_deskew):
        self.logger.debug("enter get_slopes_and_deskew_new")
        if len(contours)>15:
            num_cores = cpu_count()
        else:
            num_cores = 1
        queue_of_all_params = Queue()

        processes = []
        nh = np.linspace(0, len(boxes), num_cores + 1)
        indexes_by_text_con = np.array(range(len(contours_par)))
        for i in range(num_cores):
            boxes_per_process = boxes[int(nh[i]) : int(nh[i + 1])]
            contours_per_process = contours[int(nh[i]) : int(nh[i + 1])]
            contours_par_per_process = contours_par[int(nh[i]) : int(nh[i + 1])]
            indexes_text_con_per_process = indexes_by_text_con[int(nh[i]) : int(nh[i + 1])]

            processes.append(Process(target=self.do_work_of_slopes_new_light, args=(queue_of_all_params, boxes_per_process, textline_mask_tot, contours_per_process, contours_par_per_process, indexes_text_con_per_process, image_page_rotated, slope_deskew)))
        for i in range(num_cores):
            processes[i].start()

        slopes = []
        all_found_textline_polygons = []
        all_found_text_regions = []
        all_found_text_regions_par = []
        boxes = []
        all_box_coord = []
        all_index_text_con = []
        for i in range(num_cores):
            list_all_par = queue_of_all_params.get(True)
            slopes_for_sub_process = list_all_par[0]
            polys_for_sub_process = list_all_par[1]
            boxes_for_sub_process = list_all_par[2]
            contours_for_subprocess = list_all_par[3]
            contours_par_for_subprocess = list_all_par[4]
            boxes_coord_for_subprocess = list_all_par[5]
            indexes_for_subprocess = list_all_par[6]
            for j in range(len(slopes_for_sub_process)):
                slopes.append(slopes_for_sub_process[j])
                all_found_textline_polygons.append(polys_for_sub_process[j])
                boxes.append(boxes_for_sub_process[j])
                all_found_text_regions.append(contours_for_subprocess[j])
                all_found_text_regions_par.append(contours_par_for_subprocess[j])
                all_box_coord.append(boxes_coord_for_subprocess[j])
                all_index_text_con.append(indexes_for_subprocess[j])
        for i in range(num_cores):
            processes[i].join()
        self.logger.debug('slopes %s', slopes)
        self.logger.debug("exit get_slopes_and_deskew_new")
        return slopes, all_found_textline_polygons, boxes, all_found_text_regions, all_found_text_regions_par, all_box_coord, all_index_text_con

    def get_slopes_and_deskew_new(self, contours, contours_par, textline_mask_tot, image_page_rotated, boxes, slope_deskew):
        self.logger.debug("enter get_slopes_and_deskew_new")
        num_cores = cpu_count()
        queue_of_all_params = Queue()

        processes = []
        nh = np.linspace(0, len(boxes), num_cores + 1)
        indexes_by_text_con = np.array(range(len(contours_par)))
        for i in range(num_cores):
            boxes_per_process = boxes[int(nh[i]) : int(nh[i + 1])]
            contours_per_process = contours[int(nh[i]) : int(nh[i + 1])]
            contours_par_per_process = contours_par[int(nh[i]) : int(nh[i + 1])]
            indexes_text_con_per_process = indexes_by_text_con[int(nh[i]) : int(nh[i + 1])]

            processes.append(Process(target=self.do_work_of_slopes_new, args=(queue_of_all_params, boxes_per_process, textline_mask_tot, contours_per_process, contours_par_per_process, indexes_text_con_per_process, image_page_rotated, slope_deskew)))
        for i in range(num_cores):
            processes[i].start()

        slopes = []
        all_found_textline_polygons = []
        all_found_text_regions = []
        all_found_text_regions_par = []
        boxes = []
        all_box_coord = []
        all_index_text_con = []
        for i in range(num_cores):
            list_all_par = queue_of_all_params.get(True)
            slopes_for_sub_process = list_all_par[0]
            polys_for_sub_process = list_all_par[1]
            boxes_for_sub_process = list_all_par[2]
            contours_for_subprocess = list_all_par[3]
            contours_par_for_subprocess = list_all_par[4]
            boxes_coord_for_subprocess = list_all_par[5]
            indexes_for_subprocess = list_all_par[6]
            for j in range(len(slopes_for_sub_process)):
                slopes.append(slopes_for_sub_process[j])
                all_found_textline_polygons.append(polys_for_sub_process[j])
                boxes.append(boxes_for_sub_process[j])
                all_found_text_regions.append(contours_for_subprocess[j])
                all_found_text_regions_par.append(contours_par_for_subprocess[j])
                all_box_coord.append(boxes_coord_for_subprocess[j])
                all_index_text_con.append(indexes_for_subprocess[j])
        for i in range(num_cores):
            processes[i].join()
        self.logger.debug('slopes %s', slopes)
        self.logger.debug("exit get_slopes_and_deskew_new")
        return slopes, all_found_textline_polygons, boxes, all_found_text_regions, all_found_text_regions_par, all_box_coord, all_index_text_con

    def get_slopes_and_deskew_new_curved(self, contours, contours_par, textline_mask_tot, image_page_rotated, boxes, mask_texts_only, num_col, scale_par, slope_deskew):
        self.logger.debug("enter get_slopes_and_deskew_new_curved")
        num_cores = cpu_count()
        queue_of_all_params = Queue()

        processes = []
        nh = np.linspace(0, len(boxes), num_cores + 1)
        indexes_by_text_con = np.array(range(len(contours_par)))

        for i in range(num_cores):
            boxes_per_process = boxes[int(nh[i]) : int(nh[i + 1])]
            contours_per_process = contours[int(nh[i]) : int(nh[i + 1])]
            contours_par_per_process = contours_par[int(nh[i]) : int(nh[i + 1])]
            indexes_text_con_per_process = indexes_by_text_con[int(nh[i]) : int(nh[i + 1])]

            processes.append(Process(target=self.do_work_of_slopes_new_curved, args=(queue_of_all_params, boxes_per_process, textline_mask_tot, contours_per_process, contours_par_per_process, image_page_rotated, mask_texts_only, num_col, scale_par, indexes_text_con_per_process, slope_deskew)))

        for i in range(num_cores):
            processes[i].start()

        slopes = []
        all_found_textline_polygons = []
        all_found_text_regions = []
        all_found_text_regions_par = []
        boxes = []
        all_box_coord = []
        all_index_text_con = []

        for i in range(num_cores):
            list_all_par = queue_of_all_params.get(True)
            polys_for_sub_process = list_all_par[0]
            boxes_for_sub_process = list_all_par[1]
            contours_for_subprocess = list_all_par[2]
            contours_par_for_subprocess = list_all_par[3]
            boxes_coord_for_subprocess = list_all_par[4]
            indexes_for_subprocess = list_all_par[5]
            slopes_for_sub_process = list_all_par[6]
            for j in range(len(polys_for_sub_process)):
                slopes.append(slopes_for_sub_process[j])
                all_found_textline_polygons.append(polys_for_sub_process[j][::-1])
                boxes.append(boxes_for_sub_process[j])
                all_found_text_regions.append(contours_for_subprocess[j])
                all_found_text_regions_par.append(contours_par_for_subprocess[j])
                all_box_coord.append(boxes_coord_for_subprocess[j])
                all_index_text_con.append(indexes_for_subprocess[j])

        for i in range(num_cores):
            processes[i].join()
        # print(slopes,'slopes')
        return all_found_textline_polygons, boxes, all_found_text_regions, all_found_text_regions_par, all_box_coord, all_index_text_con, slopes

    def do_work_of_slopes_new_curved(self, queue_of_all_params, boxes_text, textline_mask_tot_ea, contours_per_process, contours_par_per_process, image_page_rotated, mask_texts_only, num_col, scale_par, indexes_r_con_per_pro, slope_deskew):
        self.logger.debug("enter do_work_of_slopes_new_curved")
        slopes_per_each_subprocess = []
        bounding_box_of_textregion_per_each_subprocess = []
        textlines_rectangles_per_each_subprocess = []
        contours_textregion_per_each_subprocess = []
        contours_textregion_par_per_each_subprocess = []
        all_box_coord_per_process = []
        index_by_text_region_contours = []

        textline_cnt_separated = np.zeros(textline_mask_tot_ea.shape)

        for mv in range(len(boxes_text)):

            all_text_region_raw = textline_mask_tot_ea[boxes_text[mv][1] : boxes_text[mv][1] + boxes_text[mv][3], boxes_text[mv][0] : boxes_text[mv][0] + boxes_text[mv][2]]
            all_text_region_raw = all_text_region_raw.astype(np.uint8)
            img_int_p = all_text_region_raw[:, :]

            # img_int_p=cv2.erode(img_int_p,KERNEL,iterations = 2)
            # plt.imshow(img_int_p)
            # plt.show()

            if img_int_p.shape[0] / img_int_p.shape[1] < 0.1:
                slopes_per_each_subprocess.append(0)
                slope_for_all = [slope_deskew][0]
            else:
                try:
                    textline_con, hierarchy = return_contours_of_image(img_int_p)
                    textline_con_fil = filter_contours_area_of_image(img_int_p, textline_con, hierarchy, max_area=1, min_area=0.0008)
                    y_diff_mean = find_contours_mean_y_diff(textline_con_fil)
                    if self.isNaN(y_diff_mean):
                        slope_for_all = MAX_SLOPE
                    else:
                        sigma_des = max(1, int(y_diff_mean * (4.0 / 40.0)))
                        img_int_p[img_int_p > 0] = 1
                        slope_for_all = return_deskew_slop(img_int_p, sigma_des, plotter=self.plotter)

                        if abs(slope_for_all) < 0.5:
                            slope_for_all = [slope_deskew][0]

                except Exception as why:
                    self.logger.error(why)
                    slope_for_all = MAX_SLOPE

                if slope_for_all == MAX_SLOPE:
                    slope_for_all = [slope_deskew][0]
                slopes_per_each_subprocess.append(slope_for_all)

            index_by_text_region_contours.append(indexes_r_con_per_pro[mv])
            _, crop_coor = crop_image_inside_box(boxes_text[mv], image_page_rotated)

            if abs(slope_for_all) < 45:
                # all_box_coord.append(crop_coor)
                textline_region_in_image = np.zeros(textline_mask_tot_ea.shape)
                cnt_o_t_max = contours_par_per_process[mv]
                x, y, w, h = cv2.boundingRect(cnt_o_t_max)
                mask_biggest = np.zeros(mask_texts_only.shape)
                mask_biggest = cv2.fillPoly(mask_biggest, pts=[cnt_o_t_max], color=(1, 1, 1))
                mask_region_in_patch_region = mask_biggest[y : y + h, x : x + w]
                textline_biggest_region = mask_biggest * textline_mask_tot_ea

                # print(slope_for_all,'slope_for_all')
                textline_rotated_separated = separate_lines_new2(textline_biggest_region[y : y + h, x : x + w], 0, num_col, slope_for_all, plotter=self.plotter)

                # new line added
                ##print(np.shape(textline_rotated_separated),np.shape(mask_biggest))
                textline_rotated_separated[mask_region_in_patch_region[:, :] != 1] = 0
                # till here

                textline_cnt_separated[y : y + h, x : x + w] = textline_rotated_separated
                textline_region_in_image[y : y + h, x : x + w] = textline_rotated_separated

                # plt.imshow(textline_region_in_image)
                # plt.show()
                # plt.imshow(textline_cnt_separated)
                # plt.show()

                pixel_img = 1
                cnt_textlines_in_image = return_contours_of_interested_textline(textline_region_in_image, pixel_img)

                textlines_cnt_per_region = []
                for jjjj in range(len(cnt_textlines_in_image)):
                    mask_biggest2 = np.zeros(mask_texts_only.shape)
                    mask_biggest2 = cv2.fillPoly(mask_biggest2, pts=[cnt_textlines_in_image[jjjj]], color=(1, 1, 1))
                    if num_col + 1 == 1:
                        mask_biggest2 = cv2.dilate(mask_biggest2, KERNEL, iterations=5)
                    else:
                        mask_biggest2 = cv2.dilate(mask_biggest2, KERNEL, iterations=4)

                    pixel_img = 1
                    mask_biggest2 = resize_image(mask_biggest2, int(mask_biggest2.shape[0] * scale_par), int(mask_biggest2.shape[1] * scale_par))
                    cnt_textlines_in_image_ind = return_contours_of_interested_textline(mask_biggest2, pixel_img)
                    try:
                        textlines_cnt_per_region.append(cnt_textlines_in_image_ind[0])
                    except Exception as why:
                        self.logger.error(why)
            else:
                add_boxes_coor_into_textlines = True
                textlines_cnt_per_region = textline_contours_postprocessing(all_text_region_raw, slope_for_all, contours_par_per_process[mv], boxes_text[mv], add_boxes_coor_into_textlines)
                add_boxes_coor_into_textlines = False
                # print(np.shape(textlines_cnt_per_region),'textlines_cnt_per_region')

            textlines_rectangles_per_each_subprocess.append(textlines_cnt_per_region)
            bounding_box_of_textregion_per_each_subprocess.append(boxes_text[mv])
            contours_textregion_per_each_subprocess.append(contours_per_process[mv])
            contours_textregion_par_per_each_subprocess.append(contours_par_per_process[mv])
            all_box_coord_per_process.append(crop_coor)

        queue_of_all_params.put([textlines_rectangles_per_each_subprocess, bounding_box_of_textregion_per_each_subprocess, contours_textregion_per_each_subprocess, contours_textregion_par_per_each_subprocess, all_box_coord_per_process, index_by_text_region_contours, slopes_per_each_subprocess])
    def do_work_of_slopes_new_light(self, queue_of_all_params, boxes_text, textline_mask_tot_ea, contours_per_process, contours_par_per_process, indexes_r_con_per_pro, image_page_rotated, slope_deskew):
        self.logger.debug('enter do_work_of_slopes_new_light')
        slopes_per_each_subprocess = []
        bounding_box_of_textregion_per_each_subprocess = []
        textlines_rectangles_per_each_subprocess = []
        contours_textregion_per_each_subprocess = []
        contours_textregion_par_per_each_subprocess = []
        all_box_coord_per_process = []
        index_by_text_region_contours = []
        for mv in range(len(boxes_text)):
            _, crop_coor = crop_image_inside_box(boxes_text[mv],image_page_rotated)
            mask_textline = np.zeros((textline_mask_tot_ea.shape))
            mask_textline = cv2.fillPoly(mask_textline,pts=[contours_per_process[mv]],color=(1,1,1))
            all_text_region_raw = (textline_mask_tot_ea*mask_textline[:,:])[boxes_text[mv][1]:boxes_text[mv][1]+boxes_text[mv][3] , boxes_text[mv][0]:boxes_text[mv][0]+boxes_text[mv][2] ]
            all_text_region_raw=all_text_region_raw.astype(np.uint8)

            slopes_per_each_subprocess.append([slope_deskew][0])
            mask_only_con_region = np.zeros(textline_mask_tot_ea.shape)
            mask_only_con_region = cv2.fillPoly(mask_only_con_region, pts=[contours_par_per_process[mv]], color=(1, 1, 1))

            
            if self.textline_light:
                all_text_region_raw = np.copy(textline_mask_tot_ea)
                all_text_region_raw[mask_only_con_region == 0] = 0
                cnt_clean_rot_raw, hir_on_cnt_clean_rot = return_contours_of_image(all_text_region_raw)
                cnt_clean_rot = filter_contours_area_of_image(all_text_region_raw, cnt_clean_rot_raw, hir_on_cnt_clean_rot, max_area=1, min_area=0.00001)
            else:
                all_text_region_raw = np.copy(textline_mask_tot_ea[boxes_text[mv][1] : boxes_text[mv][1] + boxes_text[mv][3], boxes_text[mv][0] : boxes_text[mv][0] + boxes_text[mv][2]])
                mask_only_con_region = mask_only_con_region[boxes_text[mv][1] : boxes_text[mv][1] + boxes_text[mv][3], boxes_text[mv][0] : boxes_text[mv][0] + boxes_text[mv][2]]
                all_text_region_raw[mask_only_con_region == 0] = 0
                cnt_clean_rot = textline_contours_postprocessing(all_text_region_raw, [slope_deskew][0], contours_par_per_process[mv], boxes_text[mv])

            textlines_rectangles_per_each_subprocess.append(cnt_clean_rot)
            index_by_text_region_contours.append(indexes_r_con_per_pro[mv])
            bounding_box_of_textregion_per_each_subprocess.append(boxes_text[mv])

            contours_textregion_per_each_subprocess.append(contours_per_process[mv])
            contours_textregion_par_per_each_subprocess.append(contours_par_per_process[mv])
            all_box_coord_per_process.append(crop_coor)
        queue_of_all_params.put([slopes_per_each_subprocess, textlines_rectangles_per_each_subprocess, bounding_box_of_textregion_per_each_subprocess, contours_textregion_per_each_subprocess, contours_textregion_par_per_each_subprocess, all_box_coord_per_process, index_by_text_region_contours])
        
    def do_work_of_slopes_new(self, queue_of_all_params, boxes_text, textline_mask_tot_ea, contours_per_process, contours_par_per_process, indexes_r_con_per_pro, image_page_rotated, slope_deskew):
        self.logger.debug('enter do_work_of_slopes_new')
        slopes_per_each_subprocess = []
        bounding_box_of_textregion_per_each_subprocess = []
        textlines_rectangles_per_each_subprocess = []
        contours_textregion_per_each_subprocess = []
        contours_textregion_par_per_each_subprocess = []
        all_box_coord_per_process = []
        index_by_text_region_contours = []
        for mv in range(len(boxes_text)):
            _, crop_coor = crop_image_inside_box(boxes_text[mv],image_page_rotated)
            mask_textline = np.zeros((textline_mask_tot_ea.shape))
            mask_textline = cv2.fillPoly(mask_textline,pts=[contours_per_process[mv]],color=(1,1,1))
            all_text_region_raw = (textline_mask_tot_ea*mask_textline[:,:])[boxes_text[mv][1]:boxes_text[mv][1]+boxes_text[mv][3] , boxes_text[mv][0]:boxes_text[mv][0]+boxes_text[mv][2] ]
            all_text_region_raw=all_text_region_raw.astype(np.uint8)
            img_int_p=all_text_region_raw[:,:]#self.all_text_region_raw[mv]
            img_int_p=cv2.erode(img_int_p,KERNEL,iterations = 2)

            if img_int_p.shape[0]/img_int_p.shape[1]<0.1:
                slopes_per_each_subprocess.append(0)
                slope_for_all = [slope_deskew][0]
                all_text_region_raw = textline_mask_tot_ea[boxes_text[mv][1] : boxes_text[mv][1] + boxes_text[mv][3], boxes_text[mv][0] : boxes_text[mv][0] + boxes_text[mv][2]]
                cnt_clean_rot = textline_contours_postprocessing(all_text_region_raw, slope_for_all, contours_par_per_process[mv], boxes_text[mv], 0)
                textlines_rectangles_per_each_subprocess.append(cnt_clean_rot)
                index_by_text_region_contours.append(indexes_r_con_per_pro[mv])
                bounding_box_of_textregion_per_each_subprocess.append(boxes_text[mv])
            else:
                try:
                    textline_con, hierarchy = return_contours_of_image(img_int_p)
                    textline_con_fil = filter_contours_area_of_image(img_int_p, textline_con, hierarchy, max_area=1, min_area=0.00008)
                    y_diff_mean = find_contours_mean_y_diff(textline_con_fil)
                    if self.isNaN(y_diff_mean):
                        slope_for_all = MAX_SLOPE
                    else:
                        sigma_des = int(y_diff_mean * (4.0 / 40.0))
                        if sigma_des < 1:
                            sigma_des = 1
                        img_int_p[img_int_p > 0] = 1
                        slope_for_all = return_deskew_slop(img_int_p, sigma_des, plotter=self.plotter)
                        if abs(slope_for_all) <= 0.5:
                            slope_for_all = [slope_deskew][0]
                except Exception as why:
                    self.logger.error(why)
                    slope_for_all = MAX_SLOPE
                if slope_for_all == MAX_SLOPE:
                    slope_for_all = [slope_deskew][0]
                slopes_per_each_subprocess.append(slope_for_all)
                mask_only_con_region = np.zeros(textline_mask_tot_ea.shape)
                mask_only_con_region = cv2.fillPoly(mask_only_con_region, pts=[contours_par_per_process[mv]], color=(1, 1, 1))

                # plt.imshow(mask_only_con_region)
                # plt.show()
                all_text_region_raw = np.copy(textline_mask_tot_ea[boxes_text[mv][1] : boxes_text[mv][1] + boxes_text[mv][3], boxes_text[mv][0] : boxes_text[mv][0] + boxes_text[mv][2]])
                mask_only_con_region = mask_only_con_region[boxes_text[mv][1] : boxes_text[mv][1] + boxes_text[mv][3], boxes_text[mv][0] : boxes_text[mv][0] + boxes_text[mv][2]]

                ##plt.imshow(textline_mask_tot_ea)
                ##plt.show()
                ##plt.imshow(all_text_region_raw)
                ##plt.show()
                ##plt.imshow(mask_only_con_region)
                ##plt.show()

                all_text_region_raw[mask_only_con_region == 0] = 0
                cnt_clean_rot = textline_contours_postprocessing(all_text_region_raw, slope_for_all, contours_par_per_process[mv], boxes_text[mv])

                textlines_rectangles_per_each_subprocess.append(cnt_clean_rot)
                index_by_text_region_contours.append(indexes_r_con_per_pro[mv])
                bounding_box_of_textregion_per_each_subprocess.append(boxes_text[mv])

            contours_textregion_per_each_subprocess.append(contours_per_process[mv])
            contours_textregion_par_per_each_subprocess.append(contours_par_per_process[mv])
            all_box_coord_per_process.append(crop_coor)
        queue_of_all_params.put([slopes_per_each_subprocess, textlines_rectangles_per_each_subprocess, bounding_box_of_textregion_per_each_subprocess, contours_textregion_per_each_subprocess, contours_textregion_par_per_each_subprocess, all_box_coord_per_process, index_by_text_region_contours])

    def textline_contours(self, img, patches, scaler_h, scaler_w, num_col_classifier=None):
        self.logger.debug('enter textline_contours')
        if self.textline_light:
            thresholding_for_artificial_class_in_light_version = True#False
        else:
            thresholding_for_artificial_class_in_light_version = False
        if not self.dir_in:
            model_textline, session_textline = self.start_new_session_and_model(self.model_textline_dir)
        #img = img.astype(np.uint8)
        img_org = np.copy(img)
        img_h = img_org.shape[0]
        img_w = img_org.shape[1]
        img = resize_image(img_org, int(img_org.shape[0] * scaler_h), int(img_org.shape[1] * scaler_w))
        
        if not self.dir_in:
            prediction_textline = self.do_prediction(patches, img, model_textline, marginal_of_patch_percent=0.15, n_batch_inference=3, thresholding_for_artificial_class_in_light_version=thresholding_for_artificial_class_in_light_version)
            
            #if not thresholding_for_artificial_class_in_light_version:
                #if num_col_classifier==1:
                    #prediction_textline_nopatch = self.do_prediction(False, img, model_textline)
                    #prediction_textline[:,:][prediction_textline_nopatch[:,:]==0] = 0
        else:
            prediction_textline = self.do_prediction(patches, img, self.model_textline, marginal_of_patch_percent=0.15, n_batch_inference=3,thresholding_for_artificial_class_in_light_version=thresholding_for_artificial_class_in_light_version)
            #if not thresholding_for_artificial_class_in_light_version:
                #if num_col_classifier==1:
                    #prediction_textline_nopatch = self.do_prediction(False, img, model_textline)
                    #prediction_textline[:,:][prediction_textline_nopatch[:,:]==0] = 0
        prediction_textline = resize_image(prediction_textline, img_h, img_w)
        
        textline_mask_tot_ea_art = (prediction_textline[:,:]==2)*1
        
        old_art = np.copy(textline_mask_tot_ea_art)
        
        if not thresholding_for_artificial_class_in_light_version:
            textline_mask_tot_ea_art = textline_mask_tot_ea_art.astype('uint8')
            #textline_mask_tot_ea_art = cv2.dilate(textline_mask_tot_ea_art, KERNEL, iterations=1)
            
            prediction_textline[:,:][textline_mask_tot_ea_art[:,:]==1]=2
        
        textline_mask_tot_ea_lines = (prediction_textline[:,:]==1)*1
        textline_mask_tot_ea_lines = textline_mask_tot_ea_lines.astype('uint8')
        
        if not thresholding_for_artificial_class_in_light_version:
            textline_mask_tot_ea_lines = cv2.dilate(textline_mask_tot_ea_lines, KERNEL, iterations=1)
        
        prediction_textline[:,:][textline_mask_tot_ea_lines[:,:]==1]=1
        
        if not thresholding_for_artificial_class_in_light_version:
            prediction_textline[:,:][old_art[:,:]==1]=2
        
        if not self.dir_in:
            prediction_textline_longshot = self.do_prediction(False, img, model_textline)
        else:
            prediction_textline_longshot = self.do_prediction(False, img, self.model_textline)
        prediction_textline_longshot_true_size = resize_image(prediction_textline_longshot, img_h, img_w)
        
        return ((prediction_textline[:, :, 0]==1)*1).astype('uint8'), ((prediction_textline_longshot_true_size[:, :, 0]==1)*1).astype('uint8')


    def do_work_of_slopes(self, q, poly, box_sub, boxes_per_process, textline_mask_tot, contours_per_process):
        self.logger.debug('enter do_work_of_slopes')
        slope_biggest = 0
        slopes_sub = []
        boxes_sub_new = []
        poly_sub = []
        for mv in range(len(boxes_per_process)):
            crop_img, _ = crop_image_inside_box(boxes_per_process[mv], np.repeat(textline_mask_tot[:, :, np.newaxis], 3, axis=2))
            crop_img = crop_img[:, :, 0]
            crop_img = cv2.erode(crop_img, KERNEL, iterations=2)
            try:
                textline_con, hierarchy = return_contours_of_image(crop_img)
                textline_con_fil = filter_contours_area_of_image(crop_img, textline_con, hierarchy, max_area=1, min_area=0.0008)
                y_diff_mean = find_contours_mean_y_diff(textline_con_fil)
                sigma_des = max(1, int(y_diff_mean * (4.0 / 40.0)))
                crop_img[crop_img > 0] = 1
                slope_corresponding_textregion = return_deskew_slop(crop_img, sigma_des, plotter=self.plotter)
            except Exception as why:
                self.logger.error(why)
                slope_corresponding_textregion = MAX_SLOPE

            if slope_corresponding_textregion == MAX_SLOPE:
                slope_corresponding_textregion = slope_biggest
            slopes_sub.append(slope_corresponding_textregion)

            cnt_clean_rot = textline_contours_postprocessing(crop_img, slope_corresponding_textregion, contours_per_process[mv], boxes_per_process[mv])

            poly_sub.append(cnt_clean_rot)
            boxes_sub_new.append(boxes_per_process[mv])

        q.put(slopes_sub)
        poly.put(poly_sub)
        box_sub.put(boxes_sub_new)

    def get_regions_light_v_extract_only_images(self,img,is_image_enhanced, num_col_classifier):
        self.logger.debug("enter get_regions_extract_images_only")
        erosion_hurts = False
        img_org = np.copy(img)
        img_height_h = img_org.shape[0]
        img_width_h = img_org.shape[1]

        if num_col_classifier == 1:
            img_w_new = 700
        elif num_col_classifier == 2:
            img_w_new = 900
        elif num_col_classifier == 3:
            img_w_new = 1500
        elif num_col_classifier == 4:
            img_w_new = 1800
        elif num_col_classifier == 5:
            img_w_new = 2200
        elif num_col_classifier == 6:
            img_w_new = 2500
        img_h_new = int(img.shape[0] / float(img.shape[1]) * img_w_new)

        img_resized = resize_image(img,img_h_new, img_w_new )



        if not self.dir_in:
            model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_ens_light_only_images_extraction)
            prediction_regions_org = self.do_prediction_new_concept(True, img_resized, model_region)
        else:
            prediction_regions_org = self.do_prediction_new_concept(True, img_resized, self.model_region)

        prediction_regions_org = resize_image(prediction_regions_org,img_height_h, img_width_h )

        image_page, page_coord, cont_page = self.extract_page()


        prediction_regions_org = prediction_regions_org[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]


        prediction_regions_org=prediction_regions_org[:,:,0]

        mask_lines_only = (prediction_regions_org[:,:] ==3)*1

        mask_texts_only = (prediction_regions_org[:,:] ==1)*1

        mask_images_only=(prediction_regions_org[:,:] ==2)*1

        polygons_lines_xml, hir_lines_xml = return_contours_of_image(mask_lines_only)
        polygons_lines_xml = textline_con_fil = filter_contours_area_of_image(mask_lines_only, polygons_lines_xml, hir_lines_xml, max_area=1, min_area=0.00001)


        polygons_of_only_texts = return_contours_of_interested_region(mask_texts_only,1,0.00001)

        polygons_of_only_lines = return_contours_of_interested_region(mask_lines_only,1,0.00001)

        text_regions_p_true = np.zeros(prediction_regions_org.shape)

        text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts = polygons_of_only_lines, color=(3,3,3))

        text_regions_p_true[:,:][mask_images_only[:,:] == 1] = 2

        text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts = polygons_of_only_texts, color=(1,1,1))



        text_regions_p_true[text_regions_p_true.shape[0]-15:text_regions_p_true.shape[0], :] = 0
        text_regions_p_true[:, text_regions_p_true.shape[1]-15:text_regions_p_true.shape[1]] = 0

        ##polygons_of_images = return_contours_of_interested_region(text_regions_p_true, 2, 0.0001)
        polygons_of_images = return_contours_of_interested_region(text_regions_p_true, 2, 0.001)

        image_boundary_of_doc = np.zeros((text_regions_p_true.shape[0], text_regions_p_true.shape[1]))

        ###image_boundary_of_doc[:6, :] = 1
        ###image_boundary_of_doc[text_regions_p_true.shape[0]-6:text_regions_p_true.shape[0], :] = 1

        ###image_boundary_of_doc[:, :6] = 1
        ###image_boundary_of_doc[:, text_regions_p_true.shape[1]-6:text_regions_p_true.shape[1]] = 1


        polygons_of_images_fin = []
        for ploy_img_ind in polygons_of_images:
            """
            test_poly_image = np.zeros((text_regions_p_true.shape[0], text_regions_p_true.shape[1]))
            test_poly_image = cv2.fillPoly(test_poly_image, pts = [ploy_img_ind], color=(1,1,1))
            
            test_poly_image = test_poly_image[:,:] + image_boundary_of_doc[:,:]
            test_poly_image_intersected_area = ( test_poly_image[:,:]==2 )*1
            
            test_poly_image_intersected_area = test_poly_image_intersected_area.sum()
            
            if test_poly_image_intersected_area==0:
                ##polygons_of_images_fin.append(ploy_img_ind)
                
                x, y, w, h = cv2.boundingRect(ploy_img_ind)
                box = [x, y, w, h]
                _, page_coord_img = crop_image_inside_box(box, text_regions_p_true)
                #cont_page.append(np.array([[page_coord[2], page_coord[0]], [page_coord[3], page_coord[0]], [page_coord[3], page_coord[1]], [page_coord[2], page_coord[1]]]))
                
                polygons_of_images_fin.append(np.array([[page_coord_img[2], page_coord_img[0]], [page_coord_img[3], page_coord_img[0]], [page_coord_img[3], page_coord_img[1]], [page_coord_img[2], page_coord_img[1]]]) )
            """
            x, y, w, h = cv2.boundingRect(ploy_img_ind)
            if h < 150 or w < 150:
                pass
            else:
                box = [x, y, w, h]
                _, page_coord_img = crop_image_inside_box(box, text_regions_p_true)
                #cont_page.append(np.array([[page_coord[2], page_coord[0]], [page_coord[3], page_coord[0]], [page_coord[3], page_coord[1]], [page_coord[2], page_coord[1]]]))

                polygons_of_images_fin.append(np.array([[page_coord_img[2], page_coord_img[0]], [page_coord_img[3], page_coord_img[0]], [page_coord_img[3], page_coord_img[1]], [page_coord_img[2], page_coord_img[1]]]) )

        return text_regions_p_true, erosion_hurts, polygons_lines_xml, polygons_of_images_fin, image_page, page_coord, cont_page

    def get_regions_light_v(self,img,is_image_enhanced, num_col_classifier, skip_layout_and_reading_order=False):
        self.logger.debug("enter get_regions_light_v")
        t_in = time.time()
        erosion_hurts = False
        img_org = np.copy(img)
        img_height_h = img_org.shape[0]
        img_width_h = img_org.shape[1]

        #model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_ens)

        #print(num_col_classifier,'num_col_classifier')
        
        if num_col_classifier == 1:
            img_w_new = 1000
            img_h_new = int(img_org.shape[0] / float(img_org.shape[1]) * img_w_new)
            
        elif num_col_classifier == 2:
            img_w_new = 1500#1500
            img_h_new = int(img_org.shape[0] / float(img_org.shape[1]) * img_w_new)
            
        elif num_col_classifier == 3:
            img_w_new = 2000
            img_h_new = int(img_org.shape[0] / float(img_org.shape[1]) * img_w_new)
            
        elif num_col_classifier == 4:
            img_w_new = 2500
            img_h_new = int(img_org.shape[0] / float(img_org.shape[1]) * img_w_new)
        elif num_col_classifier == 5:
            img_w_new = 3000
            img_h_new = int(img_org.shape[0] / float(img_org.shape[1]) * img_w_new)
        else:
            img_w_new = 4000
            img_h_new = int(img_org.shape[0] / float(img_org.shape[1]) * img_w_new)
        img_resized = resize_image(img,img_h_new, img_w_new )
        
        t_bin = time.time()
        
        #if (not self.input_binary) or self.full_layout:
        #if self.input_binary:
            #img_bin = np.copy(img_resized)
        ###if (not self.input_binary and self.full_layout) or (not self.input_binary and num_col_classifier >= 30):
            ###if not self.dir_in:
                ###model_bin, session_bin = self.start_new_session_and_model(self.model_dir_of_binarization)
                ###prediction_bin = self.do_prediction(True, img_resized, model_bin, n_batch_inference=5)
            ###else:
                ###prediction_bin = self.do_prediction(True, img_resized, self.model_bin, n_batch_inference=5)
                
            ####print("inside bin ", time.time()-t_bin)
            ###prediction_bin=prediction_bin[:,:,0]
            ###prediction_bin = (prediction_bin[:,:]==0)*1
            ###prediction_bin = prediction_bin*255
            
            ###prediction_bin =np.repeat(prediction_bin[:, :, np.newaxis], 3, axis=2)
            
            ###prediction_bin = prediction_bin.astype(np.uint16)
            ####img= np.copy(prediction_bin)
            ###img_bin = np.copy(prediction_bin)
        ###else:
            ###img_bin = np.copy(img_resized)
        
        img_bin = np.copy(img_resized)
        #print("inside 1 ", time.time()-t_in)
        
        ###textline_mask_tot_ea = self.run_textline(img_bin)
        textline_mask_tot_ea = self.run_textline(img_resized, num_col_classifier)
        
        
        textline_mask_tot_ea = resize_image(textline_mask_tot_ea,img_height_h, img_width_h )
        
        
        #print(self.image_org.shape)
        #cv2.imwrite('out_13.png', self.image_page_org_size)
        
        #plt.imshwo(self.image_page_org_size)
        #plt.show()
        if not skip_layout_and_reading_order:
            #print("inside 2 ", time.time()-t_in)
            if not self.dir_in:
                if num_col_classifier == 1 or num_col_classifier == 2:
                    model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_1_2_sp_np)
                    if self.image_org.shape[0]/self.image_org.shape[1] > 2.5:
                        prediction_regions_org = self.do_prediction_new_concept(True, img_resized, model_region, n_batch_inference=1, thresholding_for_some_classes_in_light_version = True)
                    else:
                        prediction_regions_org = np.zeros((self.image_org.shape[0], self.image_org.shape[1], 3))
                        prediction_regions_page = self.do_prediction_new_concept(False, self.image_page_org_size, model_region, n_batch_inference=1, thresholding_for_artificial_class_in_light_version = True)
                        prediction_regions_org[self.page_coord[0] : self.page_coord[1], self.page_coord[2] : self.page_coord[3],:] = prediction_regions_page
                else:
                    model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_1_2_sp_np)
                    prediction_regions_org = self.do_prediction_new_concept(True, resize_image(img_bin, int( (900+ (num_col_classifier-3)*100) *(img_bin.shape[0]/img_bin.shape[1]) ), 900+ (num_col_classifier-3)*100), model_region, n_batch_inference=2, thresholding_for_some_classes_in_light_version=True)
                ##model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_ens_light)
                ##prediction_regions_org = self.do_prediction(True, img_bin, model_region, n_batch_inference=3, thresholding_for_some_classes_in_light_version=True)
            else:
                if num_col_classifier == 1 or num_col_classifier == 2:
                    if self.image_org.shape[0]/self.image_org.shape[1] > 2.5:
                        prediction_regions_org = self.do_prediction_new_concept(True, img_resized, self.model_region_1_2, n_batch_inference=1, thresholding_for_some_classes_in_light_version=True)
                    else:
                        prediction_regions_org = np.zeros((self.image_org.shape[0], self.image_org.shape[1], 3))
                        prediction_regions_page = self.do_prediction_new_concept(False, self.image_page_org_size, self.model_region_1_2, n_batch_inference=1, thresholding_for_artificial_class_in_light_version=True)
                        prediction_regions_org[self.page_coord[0] : self.page_coord[1], self.page_coord[2] : self.page_coord[3],:] = prediction_regions_page
                else:
                    prediction_regions_org = self.do_prediction_new_concept(True, resize_image(img_bin, int( (900+ (num_col_classifier-3)*100) *(img_bin.shape[0]/img_bin.shape[1]) ), 900+ (num_col_classifier-3)*100), self.model_region_1_2, n_batch_inference=2, thresholding_for_some_classes_in_light_version=True)
                ###prediction_regions_org = self.do_prediction(True, img_bin, self.model_region, n_batch_inference=3, thresholding_for_some_classes_in_light_version=True)
            
            #print("inside 3 ", time.time()-t_in)
            
            #plt.imshow(prediction_regions_org[:,:,0])
            #plt.show()
            
                
            prediction_regions_org = resize_image(prediction_regions_org,img_height_h, img_width_h )
            
            img_bin = resize_image(img_bin,img_height_h, img_width_h )
            
            prediction_regions_org=prediction_regions_org[:,:,0]
            
                
            mask_lines_only = (prediction_regions_org[:,:] ==3)*1
            

            
            mask_texts_only = (prediction_regions_org[:,:] ==1)*1
            
            mask_texts_only = mask_texts_only.astype('uint8')
            
            ##if num_col_classifier == 1 or num_col_classifier == 2:
                ###mask_texts_only = cv2.erode(mask_texts_only, KERNEL, iterations=1)
                ##mask_texts_only = cv2.dilate(mask_texts_only, KERNEL, iterations=1)
            
            mask_texts_only = cv2.dilate(mask_texts_only, kernel=np.ones((2,2), np.uint8), iterations=1)
            
            
            mask_images_only=(prediction_regions_org[:,:] ==2)*1
            
            polygons_lines_xml, hir_lines_xml = return_contours_of_image(mask_lines_only)
            
            
            test_khat = np.zeros(prediction_regions_org.shape)
            
            test_khat = cv2.fillPoly(test_khat, pts = polygons_lines_xml, color=(1,1,1))
            
            
            #plt.imshow(test_khat[:,:])
            #plt.show()
            
            #for jv in range(1):
                #print(jv, hir_lines_xml[0][232][3])
                #test_khat = np.zeros(prediction_regions_org.shape)
                
                #test_khat = cv2.fillPoly(test_khat, pts = [polygons_lines_xml[232]], color=(1,1,1))
                
                
                #plt.imshow(test_khat[:,:])
                #plt.show()
                

            polygons_lines_xml = filter_contours_area_of_image(mask_lines_only, polygons_lines_xml, hir_lines_xml, max_area=1, min_area=0.00001)
            
            
            test_khat = np.zeros(prediction_regions_org.shape)
            
            test_khat = cv2.fillPoly(test_khat, pts = polygons_lines_xml, color=(1,1,1))
            
            
            #plt.imshow(test_khat[:,:])
            #plt.show()
            #sys.exit()
            
            polygons_of_only_texts = return_contours_of_interested_region(mask_texts_only,1,0.00001)
            
            ##polygons_of_only_texts = self.dilate_textregions_contours(polygons_of_only_texts)
            
            
            polygons_of_only_lines = return_contours_of_interested_region(mask_lines_only,1,0.00001)
            
            text_regions_p_true = np.zeros(prediction_regions_org.shape)
            
            text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts = polygons_of_only_lines, color=(3,3,3))
            
            text_regions_p_true[:,:][mask_images_only[:,:] == 1] = 2
            
            text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts = polygons_of_only_texts, color=(1,1,1))
            
            #plt.imshow(textline_mask_tot_ea)
            #plt.show()
            
            textline_mask_tot_ea[(text_regions_p_true==0) | (text_regions_p_true==4) ] = 0
            
            #plt.imshow(textline_mask_tot_ea)
            #plt.show()
            #print("inside 4 ", time.time()-t_in)
            return text_regions_p_true, erosion_hurts, polygons_lines_xml, textline_mask_tot_ea, img_bin
        else:
            img_bin = resize_image(img_bin,img_height_h, img_width_h )
            return None, erosion_hurts, None, textline_mask_tot_ea, img_bin

    def get_regions_from_xy_2models(self,img,is_image_enhanced, num_col_classifier):
        self.logger.debug("enter get_regions_from_xy_2models")
        erosion_hurts = False
        img_org = np.copy(img)
        img_height_h = img_org.shape[0]
        img_width_h = img_org.shape[1]
        
        if not self.dir_in:
            model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_ens)

        ratio_y=1.3
        ratio_x=1

        img = resize_image(img_org, int(img_org.shape[0]*ratio_y), int(img_org.shape[1]*ratio_x))
        if not self.dir_in:
            prediction_regions_org_y = self.do_prediction(True, img, model_region)
        else:
            prediction_regions_org_y = self.do_prediction(True, img, self.model_region)
        prediction_regions_org_y = resize_image(prediction_regions_org_y, img_height_h, img_width_h )

        #plt.imshow(prediction_regions_org_y[:,:,0])
        #plt.show()
        prediction_regions_org_y = prediction_regions_org_y[:,:,0]
        mask_zeros_y = (prediction_regions_org_y[:,:]==0)*1
        
        ##img_only_regions_with_sep = ( (prediction_regions_org_y[:,:] != 3) & (prediction_regions_org_y[:,:] != 0) )*1
        img_only_regions_with_sep = ( prediction_regions_org_y[:,:] == 1 )*1
        img_only_regions_with_sep = img_only_regions_with_sep.astype(np.uint8)
        
        try:
            img_only_regions = cv2.erode(img_only_regions_with_sep[:,:], KERNEL, iterations=20)

            _, _ = find_num_col(img_only_regions, num_col_classifier, self.tables, multiplier=6.0)
            
            img = resize_image(img_org, int(img_org.shape[0]), int(img_org.shape[1]*(1.2 if is_image_enhanced else 1)))
            
            if self.dir_in:
                prediction_regions_org = self.do_prediction(True, img, self.model_region)
            else:
                prediction_regions_org = self.do_prediction(True, img, model_region)
            prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h )

            prediction_regions_org=prediction_regions_org[:,:,0]
            prediction_regions_org[(prediction_regions_org[:,:]==1) & (mask_zeros_y[:,:]==1)]=0
            
            
            if not self.dir_in:
                model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p2)

            img = resize_image(img_org, int(img_org.shape[0]), int(img_org.shape[1]))
            
            if self.dir_in:
                prediction_regions_org2 = self.do_prediction(True, img, self.model_region_p2, marginal_of_patch_percent=0.2)
            else:
                prediction_regions_org2 = self.do_prediction(True, img, model_region, marginal_of_patch_percent=0.2)
            prediction_regions_org2=resize_image(prediction_regions_org2, img_height_h, img_width_h )


            mask_zeros2 = (prediction_regions_org2[:,:,0] == 0)
            mask_lines2 = (prediction_regions_org2[:,:,0] == 3)
            text_sume_early = (prediction_regions_org[:,:] == 1).sum()
            prediction_regions_org_copy = np.copy(prediction_regions_org)
            prediction_regions_org_copy[(prediction_regions_org_copy[:,:]==1) & (mask_zeros2[:,:]==1)] = 0
            text_sume_second = ((prediction_regions_org_copy[:,:]==1)*1).sum()

            rate_two_models = text_sume_second / float(text_sume_early) * 100

            self.logger.info("ratio_of_two_models: %s", rate_two_models)
            if not(is_image_enhanced and rate_two_models < RATIO_OF_TWO_MODEL_THRESHOLD):
                prediction_regions_org = np.copy(prediction_regions_org_copy)
                
            

            prediction_regions_org[(mask_lines2[:,:]==1) & (prediction_regions_org[:,:]==0)]=3
            mask_lines_only=(prediction_regions_org[:,:]==3)*1
            prediction_regions_org = cv2.erode(prediction_regions_org[:,:], KERNEL, iterations=2)


            prediction_regions_org = cv2.dilate(prediction_regions_org[:,:], KERNEL, iterations=2)
            
            
            if rate_two_models<=40:
                if self.input_binary:
                    prediction_bin = np.copy(img_org)
                else:
                    if not self.dir_in:
                        model_bin, session_bin = self.start_new_session_and_model(self.model_dir_of_binarization)
                        prediction_bin = self.do_prediction(True, img_org, model_bin, n_batch_inference=5)
                    else:
                        prediction_bin = self.do_prediction(True, img_org, self.model_bin, n_batch_inference=5)
                    prediction_bin = resize_image(prediction_bin, img_height_h, img_width_h )
                    
                    prediction_bin=prediction_bin[:,:,0]
                    prediction_bin = (prediction_bin[:,:]==0)*1
                    prediction_bin = prediction_bin*255
                    
                    prediction_bin =np.repeat(prediction_bin[:, :, np.newaxis], 3, axis=2)
                
                if not self.dir_in:
                    model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_ens)
                ratio_y=1
                ratio_x=1


                img = resize_image(prediction_bin, int(img_org.shape[0]*ratio_y), int(img_org.shape[1]*ratio_x))
                
                if not self.dir_in:
                    prediction_regions_org = self.do_prediction(True, img, model_region)
                else:
                    prediction_regions_org = self.do_prediction(True, img, self.model_region)
                prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h )
                prediction_regions_org=prediction_regions_org[:,:,0]
                
                mask_lines_only=(prediction_regions_org[:,:]==3)*1
                
            mask_texts_only=(prediction_regions_org[:,:]==1)*1
            mask_images_only=(prediction_regions_org[:,:]==2)*1
            
            
            
            polygons_lines_xml, hir_lines_xml = return_contours_of_image(mask_lines_only)
            polygons_lines_xml = filter_contours_area_of_image(mask_lines_only, polygons_lines_xml, hir_lines_xml, max_area=1, min_area=0.00001)

            polygons_of_only_texts = return_contours_of_interested_region(mask_texts_only, 1, 0.00001)
            polygons_of_only_lines = return_contours_of_interested_region(mask_lines_only, 1, 0.00001)

            text_regions_p_true = np.zeros(prediction_regions_org.shape)
            text_regions_p_true = cv2.fillPoly(text_regions_p_true,pts = polygons_of_only_lines, color=(3, 3, 3))
            text_regions_p_true[:,:][mask_images_only[:,:] == 1] = 2

            text_regions_p_true=cv2.fillPoly(text_regions_p_true,pts=polygons_of_only_texts, color=(1,1,1))

            return text_regions_p_true, erosion_hurts, polygons_lines_xml
        except:
            
            if self.input_binary:
                prediction_bin = np.copy(img_org)
                
                if not self.dir_in:
                    model_bin, session_bin = self.start_new_session_and_model(self.model_dir_of_binarization)
                    prediction_bin = self.do_prediction(True, img_org, model_bin, n_batch_inference=5)
                else:
                    prediction_bin = self.do_prediction(True, img_org, self.model_bin, n_batch_inference=5)
                prediction_bin = resize_image(prediction_bin, img_height_h, img_width_h )
                prediction_bin=prediction_bin[:,:,0]
                
                prediction_bin = (prediction_bin[:,:]==0)*1
                
                prediction_bin = prediction_bin*255
                
                prediction_bin =np.repeat(prediction_bin[:, :, np.newaxis], 3, axis=2)
            
            
                if not self.dir_in:
                    model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_ens)
                    
            else:
                prediction_bin = np.copy(img_org)
            ratio_y=1
            ratio_x=1


            img = resize_image(prediction_bin, int(img_org.shape[0]*ratio_y), int(img_org.shape[1]*ratio_x))
            if not self.dir_in:
                prediction_regions_org = self.do_prediction(True, img, model_region)
            else:
                prediction_regions_org = self.do_prediction(True, img, self.model_region)
            prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h )
            prediction_regions_org=prediction_regions_org[:,:,0]
            
            #mask_lines_only=(prediction_regions_org[:,:]==3)*1
            #img = resize_image(img_org, int(img_org.shape[0]*1), int(img_org.shape[1]*1))
            
            #prediction_regions_org = self.do_prediction(True, img, model_region)
            
            #prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h )
            
            #prediction_regions_org = prediction_regions_org[:,:,0]
            
            #prediction_regions_org[(prediction_regions_org[:,:] == 1) & (mask_zeros_y[:,:] == 1)]=0
            
            
            mask_lines_only = (prediction_regions_org[:,:] ==3)*1
            
            mask_texts_only = (prediction_regions_org[:,:] ==1)*1
            
            mask_images_only=(prediction_regions_org[:,:] ==2)*1
            
            polygons_lines_xml, hir_lines_xml = return_contours_of_image(mask_lines_only)
            polygons_lines_xml = filter_contours_area_of_image(mask_lines_only, polygons_lines_xml, hir_lines_xml, max_area=1, min_area=0.00001)
            
            
            polygons_of_only_texts = return_contours_of_interested_region(mask_texts_only,1,0.00001)
            
            polygons_of_only_lines = return_contours_of_interested_region(mask_lines_only,1,0.00001)
            
            
            text_regions_p_true = np.zeros(prediction_regions_org.shape)
            
            text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts = polygons_of_only_lines, color=(3,3,3))
            
            text_regions_p_true[:,:][mask_images_only[:,:] == 1] = 2
            
            text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts = polygons_of_only_texts, color=(1,1,1))
            
            erosion_hurts = True
            return text_regions_p_true, erosion_hurts, polygons_lines_xml

    def do_order_of_regions_full_layout(self, contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot):
        self.logger.debug("enter do_order_of_regions_full_layout")
        cx_text_only, cy_text_only, x_min_text_only, _, _, _, y_cor_x_min_main = find_new_features_of_contours(contours_only_text_parent)
        cx_text_only_h, cy_text_only_h, x_min_text_only_h, _, _, _, y_cor_x_min_main_h = find_new_features_of_contours(contours_only_text_parent_h)

        try:
            arg_text_con = []
            for ii in range(len(cx_text_only)):
                for jj in range(len(boxes)):
                    if (x_min_text_only[ii] + 80) >= boxes[jj][0] and (x_min_text_only[ii] + 80) < boxes[jj][1] and y_cor_x_min_main[ii] >= boxes[jj][2] and y_cor_x_min_main[ii] < boxes[jj][3]:
                        arg_text_con.append(jj)
                        break
            args_contours = np.array(range(len(arg_text_con)))
            arg_text_con_h = []
            for ii in range(len(cx_text_only_h)):
                for jj in range(len(boxes)):
                    if (x_min_text_only_h[ii] + 80) >= boxes[jj][0] and (x_min_text_only_h[ii] + 80) < boxes[jj][1] and y_cor_x_min_main_h[ii] >= boxes[jj][2] and y_cor_x_min_main_h[ii] < boxes[jj][3]:
                        arg_text_con_h.append(jj)
                        break
            args_contours_h = np.array(range(len(arg_text_con_h)))

            order_by_con_head = np.zeros(len(arg_text_con_h))
            order_by_con_main = np.zeros(len(arg_text_con))

            ref_point = 0
            order_of_texts_tot = []
            id_of_texts_tot = []
            for iij in range(len(boxes)):

                args_contours_box = args_contours[np.array(arg_text_con) == iij]
                args_contours_box_h = args_contours_h[np.array(arg_text_con_h) == iij]
                con_inter_box = []
                con_inter_box_h = []

                for box in args_contours_box:
                    con_inter_box.append(contours_only_text_parent[box])

                for box in args_contours_box_h:
                    con_inter_box_h.append(contours_only_text_parent_h[box])

                indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = order_of_regions(textline_mask_tot[int(boxes[iij][2]) : int(boxes[iij][3]), int(boxes[iij][0]) : int(boxes[iij][1])], con_inter_box, con_inter_box_h, boxes[iij][2])

                order_of_texts, id_of_texts = order_and_id_of_texts(con_inter_box, con_inter_box_h, matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

                indexes_sorted_main = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_by_type_main = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_sorted_head = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 2]
                indexes_by_type_head = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 2]

                for zahler, _ in enumerate(args_contours_box):
                    arg_order_v = indexes_sorted_main[zahler]
                    order_by_con_main[args_contours_box[indexes_by_type_main[zahler]]] = np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

                for zahler, _ in enumerate(args_contours_box_h):
                    arg_order_v = indexes_sorted_head[zahler]
                    order_by_con_head[args_contours_box_h[indexes_by_type_head[zahler]]] = np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

                for jji in range(len(id_of_texts)):
                    order_of_texts_tot.append(order_of_texts[jji] + ref_point)
                    id_of_texts_tot.append(id_of_texts[jji])
                ref_point += len(id_of_texts)

            order_of_texts_tot = []
            for tj1 in range(len(contours_only_text_parent)):
                order_of_texts_tot.append(int(order_by_con_main[tj1]))

            for tj1 in range(len(contours_only_text_parent_h)):
                order_of_texts_tot.append(int(order_by_con_head[tj1]))

            order_text_new = []
            for iii in range(len(order_of_texts_tot)):
                order_text_new.append(np.where(np.array(order_of_texts_tot) == iii)[0][0])

        except Exception as why:
            self.logger.error(why)
            arg_text_con = []
            for ii in range(len(cx_text_only)):
                for jj in range(len(boxes)):
                    if cx_text_only[ii] >= boxes[jj][0] and cx_text_only[ii] < boxes[jj][1] and cy_text_only[ii] >= boxes[jj][2] and cy_text_only[ii] < boxes[jj][3]:  # this is valid if the center of region identify in which box it is located
                        arg_text_con.append(jj)
                        break
            args_contours = np.array(range(len(arg_text_con)))

            order_by_con_main = np.zeros(len(arg_text_con))

            ############################# head

            arg_text_con_h = []
            for ii in range(len(cx_text_only_h)):
                for jj in range(len(boxes)):
                    if cx_text_only_h[ii] >= boxes[jj][0] and cx_text_only_h[ii] < boxes[jj][1] and cy_text_only_h[ii] >= boxes[jj][2] and cy_text_only_h[ii] < boxes[jj][3]:  # this is valid if the center of region identify in which box it is located
                        arg_text_con_h.append(jj)
                        break
            args_contours_h = np.array(range(len(arg_text_con_h)))

            order_by_con_head = np.zeros(len(arg_text_con_h))

            ref_point = 0
            order_of_texts_tot = []
            id_of_texts_tot = []
            for iij, _ in enumerate(boxes):
                args_contours_box = args_contours[np.array(arg_text_con) == iij]
                args_contours_box_h = args_contours_h[np.array(arg_text_con_h) == iij]
                con_inter_box = []
                con_inter_box_h = []

                for box in args_contours_box:
                    con_inter_box.append(contours_only_text_parent[box])

                for box in args_contours_box_h:
                    con_inter_box_h.append(contours_only_text_parent_h[box])

                indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = order_of_regions(textline_mask_tot[int(boxes[iij][2]) : int(boxes[iij][3]), int(boxes[iij][0]) : int(boxes[iij][1])], con_inter_box, con_inter_box_h, boxes[iij][2])

                order_of_texts, id_of_texts = order_and_id_of_texts(con_inter_box, con_inter_box_h, matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

                indexes_sorted_main = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_by_type_main = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_sorted_head = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 2]
                indexes_by_type_head = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 2]

                for zahler, _ in enumerate(args_contours_box):
                    arg_order_v = indexes_sorted_main[zahler]
                    order_by_con_main[args_contours_box[indexes_by_type_main[zahler]]] = np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

                for zahler, _ in enumerate(args_contours_box_h):
                    arg_order_v = indexes_sorted_head[zahler]
                    order_by_con_head[args_contours_box_h[indexes_by_type_head[zahler]]] = np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

                for jji, _ in enumerate(id_of_texts):
                    order_of_texts_tot.append(order_of_texts[jji] + ref_point)
                    id_of_texts_tot.append(id_of_texts[jji])
                ref_point += len(id_of_texts)

            order_of_texts_tot = []
            for tj1 in range(len(contours_only_text_parent)):
                order_of_texts_tot.append(int(order_by_con_main[tj1]))

            for tj1 in range(len(contours_only_text_parent_h)):
                order_of_texts_tot.append(int(order_by_con_head[tj1]))

            order_text_new = []
            for iii in range(len(order_of_texts_tot)):
                order_text_new.append(np.where(np.array(order_of_texts_tot) == iii)[0][0])
        return order_text_new, id_of_texts_tot

    def do_order_of_regions_no_full_layout(self, contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot):
        self.logger.debug("enter do_order_of_regions_no_full_layout")
        cx_text_only, cy_text_only, x_min_text_only, _, _, _, y_cor_x_min_main = find_new_features_of_contours(contours_only_text_parent)

        try:
            arg_text_con = []
            for ii in range(len(cx_text_only)):
                for jj in range(len(boxes)):
                    if (x_min_text_only[ii] + 80) >= boxes[jj][0] and (x_min_text_only[ii] + 80) < boxes[jj][1] and y_cor_x_min_main[ii] >= boxes[jj][2] and y_cor_x_min_main[ii] < boxes[jj][3]:
                        arg_text_con.append(jj)
                        break
            args_contours = np.array(range(len(arg_text_con)))
            order_by_con_main = np.zeros(len(arg_text_con))

            ref_point = 0
            order_of_texts_tot = []
            id_of_texts_tot = []
            for iij in range(len(boxes)):
                args_contours_box = args_contours[np.array(arg_text_con) == iij]
                con_inter_box = []
                con_inter_box_h = []
                for i in range(len(args_contours_box)):
                    con_inter_box.append(contours_only_text_parent[args_contours_box[i]])

                indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = order_of_regions(textline_mask_tot[int(boxes[iij][2]) : int(boxes[iij][3]), int(boxes[iij][0]) : int(boxes[iij][1])], con_inter_box, con_inter_box_h, boxes[iij][2])

                order_of_texts, id_of_texts = order_and_id_of_texts(con_inter_box, con_inter_box_h, matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

                indexes_sorted_main = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_by_type_main = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 1]

                for zahler, _ in enumerate(args_contours_box):
                    arg_order_v = indexes_sorted_main[zahler]
                    order_by_con_main[args_contours_box[indexes_by_type_main[zahler]]] = np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

                for jji, _ in enumerate(id_of_texts):
                    order_of_texts_tot.append(order_of_texts[jji] + ref_point)
                    id_of_texts_tot.append(id_of_texts[jji])
                ref_point += len(id_of_texts)

            order_of_texts_tot = []
            for tj1 in range(len(contours_only_text_parent)):
                order_of_texts_tot.append(int(order_by_con_main[tj1]))

            order_text_new = []
            for iii in range(len(order_of_texts_tot)):
                order_text_new.append(np.where(np.array(order_of_texts_tot) == iii)[0][0])
        
        except Exception as why:
            self.logger.error(why)
            arg_text_con = []
            for ii in range(len(cx_text_only)):
                for jj in range(len(boxes)):
                    if cx_text_only[ii] >= boxes[jj][0] and cx_text_only[ii] < boxes[jj][1] and cy_text_only[ii] >= boxes[jj][2] and cy_text_only[ii] < boxes[jj][3]:  # this is valid if the center of region identify in which box it is located
                        arg_text_con.append(jj)
                        break
            args_contours = np.array(range(len(arg_text_con)))

            order_by_con_main = np.zeros(len(arg_text_con))

            ref_point = 0
            order_of_texts_tot = []
            id_of_texts_tot = []
            for iij in range(len(boxes)):
                args_contours_box = args_contours[np.array(arg_text_con) == iij]
                con_inter_box = []
                con_inter_box_h = []

                for i in range(len(args_contours_box)):
                    con_inter_box.append(contours_only_text_parent[args_contours_box[i]])

                indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = order_of_regions(textline_mask_tot[int(boxes[iij][2]) : int(boxes[iij][3]), int(boxes[iij][0]) : int(boxes[iij][1])], con_inter_box, con_inter_box_h, boxes[iij][2])

                order_of_texts, id_of_texts = order_and_id_of_texts(con_inter_box, con_inter_box_h, matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

                indexes_sorted_main = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_by_type_main = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 1]

                for zahler, _ in enumerate(args_contours_box):
                    arg_order_v = indexes_sorted_main[zahler]
                    order_by_con_main[args_contours_box[indexes_by_type_main[zahler]]] = np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

                for jji, _ in enumerate(id_of_texts):
                    order_of_texts_tot.append(order_of_texts[jji] + ref_point)
                    id_of_texts_tot.append(id_of_texts[jji])
                ref_point += len(id_of_texts)

            order_of_texts_tot = []
            
            for tj1 in range(len(contours_only_text_parent)):
                order_of_texts_tot.append(int(order_by_con_main[tj1]))

            order_text_new = []
            for iii in range(len(order_of_texts_tot)):
                order_text_new.append(np.where(np.array(order_of_texts_tot) == iii)[0][0])
        
        return order_text_new, id_of_texts_tot
    def check_iou_of_bounding_box_and_contour_for_tables(self, layout, table_prediction_early, pixel_tabel, num_col_classifier):
        layout_org  = np.copy(layout)
        layout_org[:,:,0][layout_org[:,:,0]==pixel_tabel] = 0
        layout = (layout[:,:,0]==pixel_tabel)*1

        layout =np.repeat(layout[:, :, np.newaxis], 3, axis=2)
        layout = layout.astype(np.uint8)
        imgray = cv2.cvtColor(layout, cv2.COLOR_BGR2GRAY )
        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])
        
        contours_new = []
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            iou = cnt_size[i] /float(w*h) *100
            
            if iou<80:
                layout_contour = np.zeros((layout_org.shape[0], layout_org.shape[1]))
                layout_contour= cv2.fillPoly(layout_contour,pts=[contours[i]] ,color=(1,1,1))
                
                
                layout_contour_sum = layout_contour.sum(axis=0)
                layout_contour_sum_diff = np.diff(layout_contour_sum)
                layout_contour_sum_diff= np.abs(layout_contour_sum_diff)
                layout_contour_sum_diff_smoothed= gaussian_filter1d(layout_contour_sum_diff, 10)

                peaks, _ = find_peaks(layout_contour_sum_diff_smoothed, height=0)
                peaks= peaks[layout_contour_sum_diff_smoothed[peaks]>4]
                
                for j in range(len(peaks)):
                    layout_contour[:,peaks[j]-3+1:peaks[j]+1+3] = 0
                    
                layout_contour=cv2.erode(layout_contour[:,:], KERNEL, iterations=5)
                layout_contour=cv2.dilate(layout_contour[:,:], KERNEL, iterations=5)
                
                layout_contour =np.repeat(layout_contour[:, :, np.newaxis], 3, axis=2)
                layout_contour = layout_contour.astype(np.uint8)
                
                imgray = cv2.cvtColor(layout_contour, cv2.COLOR_BGR2GRAY )
                _, thresh = cv2.threshold(imgray, 0, 255, 0)

                contours_sep, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for ji in range(len(contours_sep) ):
                    contours_new.append(contours_sep[ji])
                    if num_col_classifier>=2:
                        only_recent_contour_image = np.zeros((layout.shape[0],layout.shape[1]))
                        only_recent_contour_image= cv2.fillPoly(only_recent_contour_image,pts=[contours_sep[ji]] ,color=(1,1,1))
                        table_pixels_masked_from_early_pre = only_recent_contour_image[:,:]*table_prediction_early[:,:]
                        iou_in = table_pixels_masked_from_early_pre.sum() /float(only_recent_contour_image.sum()) *100
                        #print(iou_in,'iou_in_in1')
                        
                        if iou_in>30:
                            layout_org= cv2.fillPoly(layout_org,pts=[contours_sep[ji]] ,color=(pixel_tabel,pixel_tabel,pixel_tabel))
                        else:
                            pass
                    else:
                        
                        layout_org= cv2.fillPoly(layout_org,pts=[contours_sep[ji]] ,color=(pixel_tabel,pixel_tabel,pixel_tabel))
                
            else:
                contours_new.append(contours[i])
                if num_col_classifier>=2:
                    only_recent_contour_image = np.zeros((layout.shape[0],layout.shape[1]))
                    only_recent_contour_image= cv2.fillPoly(only_recent_contour_image,pts=[contours[i]] ,color=(1,1,1))
                    
                    table_pixels_masked_from_early_pre = only_recent_contour_image[:,:]*table_prediction_early[:,:]
                    iou_in = table_pixels_masked_from_early_pre.sum() /float(only_recent_contour_image.sum()) *100
                    #print(iou_in,'iou_in')
                    if iou_in>30:
                        layout_org= cv2.fillPoly(layout_org,pts=[contours[i]] ,color=(pixel_tabel,pixel_tabel,pixel_tabel))
                    else:
                        pass
                else:
                    layout_org= cv2.fillPoly(layout_org,pts=[contours[i]] ,color=(pixel_tabel,pixel_tabel,pixel_tabel))
                
        return layout_org, contours_new
    def delete_separator_around(self,spliter_y,peaks_neg,image_by_region, pixel_line, pixel_table):
        # format of subboxes: box=[x1, x2 , y1, y2]
        pix_del = 100
        if len(image_by_region.shape)==3:
            for i in range(len(spliter_y)-1):
                for j in range(1,len(peaks_neg[i])-1):
                    image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-pix_del:peaks_neg[i][j]+pix_del,0][image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-pix_del:peaks_neg[i][j]+pix_del,0]==pixel_line ]=0
                    image_by_region[spliter_y[i]:spliter_y[i+1],peaks_neg[i][j]-pix_del:peaks_neg[i][j]+pix_del,0][image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-pix_del:peaks_neg[i][j]+pix_del,1]==pixel_line ]=0
                    image_by_region[spliter_y[i]:spliter_y[i+1],peaks_neg[i][j]-pix_del:peaks_neg[i][j]+pix_del,0][image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-pix_del:peaks_neg[i][j]+pix_del,2]==pixel_line ]=0
                    
                    image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-pix_del:peaks_neg[i][j]+pix_del,0][image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-pix_del:peaks_neg[i][j]+pix_del,0]==pixel_table ]=0
                    image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-pix_del:peaks_neg[i][j]+pix_del,0][image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-pix_del:peaks_neg[i][j]+pix_del,1]==pixel_table ]=0
                    image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-pix_del:peaks_neg[i][j]+pix_del,0][image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-pix_del:peaks_neg[i][j]+pix_del,2]==pixel_table ]=0
        else:
            for i in range(len(spliter_y)-1):
                for j in range(1,len(peaks_neg[i])-1):
                    image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-pix_del:peaks_neg[i][j]+pix_del][image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-pix_del:peaks_neg[i][j]+pix_del]==pixel_line ]=0
                    
                    image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-pix_del:peaks_neg[i][j]+pix_del][image_by_region[int(spliter_y[i]):int(spliter_y[i+1]),peaks_neg[i][j]-pix_del:peaks_neg[i][j]+pix_del]==pixel_table ]=0
        return image_by_region
    def add_tables_heuristic_to_layout(self, image_regions_eraly_p,boxes, slope_mean_hor, spliter_y,peaks_neg_tot, image_revised, num_col_classifier, min_area, pixel_line):
        pixel_table =10
        image_revised_1 = self.delete_separator_around(spliter_y, peaks_neg_tot, image_revised, pixel_line, pixel_table)
        
        try:
            image_revised_1[:,:30][image_revised_1[:,:30]==pixel_line] = 0
            image_revised_1[:,image_revised_1.shape[1]-30:][image_revised_1[:,image_revised_1.shape[1]-30:]==pixel_line] = 0
        except:
            pass
        
        img_comm_e = np.zeros(image_revised_1.shape)
        img_comm = np.repeat(img_comm_e[:, :, np.newaxis], 3, axis=2)

        for indiv in np.unique(image_revised_1):
            image_col=(image_revised_1==indiv)*255
            img_comm_in=np.repeat(image_col[:, :, np.newaxis], 3, axis=2)
            img_comm_in=img_comm_in.astype(np.uint8)

            imgray = cv2.cvtColor(img_comm_in, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 0, 255, 0)
            contours,hirarchy=cv2.findContours(thresh.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            if indiv==pixel_table:
                main_contours = filter_contours_area_of_image_tables(thresh, contours, hirarchy, max_area = 1, min_area = 0.001)
            else:
                main_contours = filter_contours_area_of_image_tables(thresh, contours, hirarchy, max_area = 1, min_area = min_area)

            img_comm = cv2.fillPoly(img_comm, pts = main_contours, color = (indiv, indiv, indiv))
            img_comm = img_comm.astype(np.uint8)
            
        if not self.isNaN(slope_mean_hor):
            image_revised_last = np.zeros((image_regions_eraly_p.shape[0], image_regions_eraly_p.shape[1],3))
            for i in range(len(boxes)):
                image_box=img_comm[int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][0]):int(boxes[i][1]),:]
                try:
                    image_box_tabels_1=(image_box[:,:,0]==pixel_table)*1
                    contours_tab,_=return_contours_of_image(image_box_tabels_1)
                    contours_tab=filter_contours_area_of_image_tables(image_box_tabels_1,contours_tab,_,1,0.003)
                    image_box_tabels_1=(image_box[:,:,0]==pixel_line)*1

                    image_box_tabels_and_m_text=( (image_box[:,:,0]==pixel_table) | (image_box[:,:,0]==1) )*1
                    image_box_tabels_and_m_text=image_box_tabels_and_m_text.astype(np.uint8)

                    image_box_tabels_1=image_box_tabels_1.astype(np.uint8)
                    image_box_tabels_1 = cv2.dilate(image_box_tabels_1,KERNEL,iterations = 5)

                    contours_table_m_text,_=return_contours_of_image(image_box_tabels_and_m_text)
                    image_box_tabels=np.repeat(image_box_tabels_1[:, :, np.newaxis], 3, axis=2)

                    image_box_tabels=image_box_tabels.astype(np.uint8)
                    imgray = cv2.cvtColor(image_box_tabels, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                    contours_line,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                    y_min_main_line ,y_max_main_line=find_features_of_contours(contours_line)
                    y_min_main_tab ,y_max_main_tab=find_features_of_contours(contours_tab)

                    cx_tab_m_text,cy_tab_m_text ,x_min_tab_m_text , x_max_tab_m_text, y_min_tab_m_text ,y_max_tab_m_text, _= find_new_features_of_contours(contours_table_m_text)
                    cx_tabl,cy_tabl ,x_min_tabl , x_max_tabl, y_min_tabl ,y_max_tabl,_= find_new_features_of_contours(contours_tab)

                    if len(y_min_main_tab )>0:
                        y_down_tabs=[]
                        y_up_tabs=[]

                        for i_t in range(len(y_min_main_tab )):
                            y_down_tab=[]
                            y_up_tab=[]
                            for i_l in range(len(y_min_main_line)):
                                if y_min_main_tab[i_t]>y_min_main_line[i_l] and  y_max_main_tab[i_t]>y_min_main_line[i_l] and y_min_main_tab[i_t]>y_max_main_line[i_l] and y_max_main_tab[i_t]>y_min_main_line[i_l]:
                                    pass
                                elif y_min_main_tab[i_t]<y_max_main_line[i_l] and y_max_main_tab[i_t]<y_max_main_line[i_l] and y_max_main_tab[i_t]<y_min_main_line[i_l] and y_min_main_tab[i_t]<y_min_main_line[i_l]:
                                    pass
                                elif np.abs(y_max_main_line[i_l]-y_min_main_line[i_l])<100:
                                    pass
                                else:
                                    y_up_tab.append(np.min([y_min_main_line[i_l], y_min_main_tab[i_t] ])  )
                                    y_down_tab.append( np.max([ y_max_main_line[i_l],y_max_main_tab[i_t] ]) )

                            if len(y_up_tab)==0:
                                y_up_tabs.append(y_min_main_tab[i_t])
                                y_down_tabs.append(y_max_main_tab[i_t])
                            else:
                                y_up_tabs.append(np.min(y_up_tab))
                                y_down_tabs.append(np.max(y_down_tab))
                    else:
                        y_down_tabs=[]
                        y_up_tabs=[]
                        pass
                except:
                    y_down_tabs=[]
                    y_up_tabs=[]

                for ii in range(len(y_up_tabs)):
                    image_box[y_up_tabs[ii]:y_down_tabs[ii],:,0]=pixel_table

                image_revised_last[int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][0]):int(boxes[i][1]),:]=image_box[:,:,:]
        else:
            for i in range(len(boxes)):

                image_box=img_comm[int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][0]):int(boxes[i][1]),:]
                image_revised_last[int(boxes[i][2]):int(boxes[i][3]),int(boxes[i][0]):int(boxes[i][1]),:]=image_box[:,:,:]
        
        if num_col_classifier==1:
            img_tables_col_1=( image_revised_last[:,:,0]==pixel_table )*1
            img_tables_col_1=img_tables_col_1.astype(np.uint8)
            contours_table_col1,_=return_contours_of_image(img_tables_col_1)
            
            _,_ ,_ , _, y_min_tab_col1 ,y_max_tab_col1, _= find_new_features_of_contours(contours_table_col1)
            
            if len(y_min_tab_col1)>0:
                for ijv in range(len(y_min_tab_col1)):
                    image_revised_last[int(y_min_tab_col1[ijv]):int(y_max_tab_col1[ijv]),:,:]=pixel_table
        return image_revised_last
    def do_order_of_regions(self, *args, **kwargs):
        if self.full_layout:
            return self.do_order_of_regions_full_layout(*args, **kwargs)
        return self.do_order_of_regions_no_full_layout(*args, **kwargs)
    
    def get_tables_from_model(self, img, num_col_classifier):
        img_org = np.copy(img)
        
        img_height_h = img_org.shape[0]
        img_width_h = img_org.shape[1]
        
        model_region, session_region = self.start_new_session_and_model(self.model_tables)
        
        patches = False
        
        if num_col_classifier < 4 and num_col_classifier > 2:
            prediction_table = self.do_prediction(patches, img, model_region)
            pre_updown = self.do_prediction(patches, cv2.flip(img[:,:,:], -1), model_region)
            pre_updown = cv2.flip(pre_updown, -1)
            
            prediction_table[:,:,0][pre_updown[:,:,0]==1]=1
            prediction_table = prediction_table.astype(np.int16)
            
        elif num_col_classifier ==2:
            height_ext = 0#int( img.shape[0]/4. )
            h_start = int(height_ext/2.)
            width_ext = int( img.shape[1]/8. )
            w_start = int(width_ext/2.)
        
            height_new = img.shape[0]+height_ext
            width_new = img.shape[1]+width_ext
            
            img_new =np.ones((height_new,width_new,img.shape[2])).astype(float)*0
            img_new[h_start:h_start+img.shape[0] ,w_start: w_start+img.shape[1], : ] =img[:,:,:]

            prediction_ext = self.do_prediction(patches, img_new, model_region)
            pre_updown = self.do_prediction(patches, cv2.flip(img_new[:,:,:], -1), model_region)
            pre_updown = cv2.flip(pre_updown, -1)
            
            prediction_table = prediction_ext[h_start:h_start+img.shape[0] ,w_start: w_start+img.shape[1], : ]
            prediction_table_updown = pre_updown[h_start:h_start+img.shape[0] ,w_start: w_start+img.shape[1], : ]
            
            prediction_table[:,:,0][prediction_table_updown[:,:,0]==1]=1
            prediction_table = prediction_table.astype(np.int16)

        elif num_col_classifier ==1:
            height_ext = 0# int( img.shape[0]/4. )
            h_start = int(height_ext/2.)
            width_ext = int( img.shape[1]/4. )
            w_start = int(width_ext/2.)
        
            height_new = img.shape[0]+height_ext
            width_new = img.shape[1]+width_ext
            
            img_new =np.ones((height_new,width_new,img.shape[2])).astype(float)*0
            img_new[h_start:h_start+img.shape[0] ,w_start: w_start+img.shape[1], : ] =img[:,:,:]

            prediction_ext = self.do_prediction(patches, img_new, model_region)
            pre_updown = self.do_prediction(patches, cv2.flip(img_new[:,:,:], -1), model_region)
            pre_updown = cv2.flip(pre_updown, -1)
            
            prediction_table = prediction_ext[h_start:h_start+img.shape[0] ,w_start: w_start+img.shape[1], : ]
            prediction_table_updown = pre_updown[h_start:h_start+img.shape[0] ,w_start: w_start+img.shape[1], : ]
            
            prediction_table[:,:,0][prediction_table_updown[:,:,0]==1]=1
            prediction_table = prediction_table.astype(np.int16)

        else:
            prediction_table = np.zeros(img.shape)
            img_w_half = int(img.shape[1]/2.)

            pre1 = self.do_prediction(patches, img[:,0:img_w_half,:], model_region)
            pre2 = self.do_prediction(patches, img[:,img_w_half:,:], model_region)
            pre_full = self.do_prediction(patches, img[:,:,:], model_region)
            pre_updown = self.do_prediction(patches, cv2.flip(img[:,:,:], -1), model_region)
            pre_updown = cv2.flip(pre_updown, -1)
            
            prediction_table_full_erode = cv2.erode(pre_full[:,:,0], KERNEL, iterations=4)
            prediction_table_full_erode = cv2.dilate(prediction_table_full_erode, KERNEL, iterations=4)
            
            prediction_table_full_updown_erode = cv2.erode(pre_updown[:,:,0], KERNEL, iterations=4)
            prediction_table_full_updown_erode = cv2.dilate(prediction_table_full_updown_erode, KERNEL, iterations=4)

            prediction_table[:,0:img_w_half,:] = pre1[:,:,:]
            prediction_table[:,img_w_half:,:] = pre2[:,:,:]
            
            prediction_table[:,:,0][prediction_table_full_erode[:,:]==1]=1
            prediction_table[:,:,0][prediction_table_full_updown_erode[:,:]==1]=1
            prediction_table = prediction_table.astype(np.int16)
            
        #prediction_table_erode = cv2.erode(prediction_table[:,:,0], self.kernel, iterations=6)
        #prediction_table_erode = cv2.dilate(prediction_table_erode, self.kernel, iterations=6)
        
        prediction_table_erode = cv2.erode(prediction_table[:,:,0], KERNEL, iterations=20)
        prediction_table_erode = cv2.dilate(prediction_table_erode, KERNEL, iterations=20)
        return prediction_table_erode.astype(np.int16)

    def run_graphics_and_columns_light(self, text_regions_p_1, textline_mask_tot_ea, num_col_classifier, num_column_is_classified, erosion_hurts, img_bin_light):
        #print(text_regions_p_1.shape, 'text_regions_p_1 shape run graphics')
        #print(erosion_hurts, 'erosion_hurts')
        t_in_gr = time.time()
        img_g = self.imread(grayscale=True, uint8=True)

        img_g3 = np.zeros((img_g.shape[0], img_g.shape[1], 3))
        img_g3 = img_g3.astype(np.uint8)
        img_g3[:, :, 0] = img_g[:, :]
        img_g3[:, :, 1] = img_g[:, :]
        img_g3[:, :, 2] = img_g[:, :]

        image_page, page_coord, cont_page = self.extract_page()
        #print("inside graphics 1 ", time.time() - t_in_gr)
        if self.tables:
            table_prediction = self.get_tables_from_model(image_page, num_col_classifier)
        else:
            table_prediction = (np.zeros((image_page.shape[0], image_page.shape[1]))).astype(np.int16)
        
        if self.plotter:
            self.plotter.save_page_image(image_page)

        text_regions_p_1 = text_regions_p_1[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        textline_mask_tot_ea = textline_mask_tot_ea[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        
        img_bin_light = img_bin_light[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        
        mask_images = (text_regions_p_1[:, :] == 2) * 1
        mask_images = mask_images.astype(np.uint8)
        mask_images = cv2.erode(mask_images[:, :], KERNEL, iterations=10)
        mask_lines = (text_regions_p_1[:, :] == 3) * 1
        mask_lines = mask_lines.astype(np.uint8)
        img_only_regions_with_sep = ((text_regions_p_1[:, :] != 3) & (text_regions_p_1[:, :] != 0)) * 1
        img_only_regions_with_sep = img_only_regions_with_sep.astype(np.uint8)
        
        #print("inside graphics 2 ", time.time() - t_in_gr)
        if erosion_hurts:
            img_only_regions = np.copy(img_only_regions_with_sep[:,:])
        else:
            img_only_regions = cv2.erode(img_only_regions_with_sep[:,:], KERNEL, iterations=6)
        
        ##print(img_only_regions.shape,'img_only_regions')
        ##plt.imshow(img_only_regions[:,:])
        ##plt.show()
        ##num_col, _ = find_num_col(img_only_regions, num_col_classifier, self.tables, multiplier=6.0)
        try:
            num_col, _ = find_num_col(img_only_regions, num_col_classifier, self.tables, multiplier=6.0)
            num_col = num_col + 1
            if not num_column_is_classified:
                num_col_classifier = num_col + 1
        except Exception as why:
            self.logger.error(why)
            num_col = None
        #print("inside graphics 3 ", time.time() - t_in_gr)
        return num_col, num_col_classifier, img_only_regions, page_coord, image_page, mask_images, mask_lines, text_regions_p_1, cont_page, table_prediction, textline_mask_tot_ea, img_bin_light
    
    def run_graphics_and_columns_without_layout(self, textline_mask_tot_ea, img_bin_light):
        
        #print(text_regions_p_1.shape, 'text_regions_p_1 shape run graphics')
        #print(erosion_hurts, 'erosion_hurts')
        t_in_gr = time.time()
        img_g = self.imread(grayscale=True, uint8=True)

        img_g3 = np.zeros((img_g.shape[0], img_g.shape[1], 3))
        img_g3 = img_g3.astype(np.uint8)
        img_g3[:, :, 0] = img_g[:, :]
        img_g3[:, :, 1] = img_g[:, :]
        img_g3[:, :, 2] = img_g[:, :]

        image_page, page_coord, cont_page = self.extract_page()
        #print("inside graphics 1 ", time.time() - t_in_gr)
        
        textline_mask_tot_ea = textline_mask_tot_ea[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        
        img_bin_light = img_bin_light[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        
        return  page_coord, image_page, textline_mask_tot_ea, img_bin_light, cont_page
    def run_graphics_and_columns(self, text_regions_p_1, num_col_classifier, num_column_is_classified, erosion_hurts):
        t_in_gr = time.time()
        img_g = self.imread(grayscale=True, uint8=True)

        img_g3 = np.zeros((img_g.shape[0], img_g.shape[1], 3))
        img_g3 = img_g3.astype(np.uint8)
        img_g3[:, :, 0] = img_g[:, :]
        img_g3[:, :, 1] = img_g[:, :]
        img_g3[:, :, 2] = img_g[:, :]

        image_page, page_coord, cont_page = self.extract_page()
        
        if self.tables:
            table_prediction = self.get_tables_from_model(image_page, num_col_classifier)
        else:
            table_prediction = (np.zeros((image_page.shape[0], image_page.shape[1]))).astype(np.int16)
        
        if self.plotter:
            self.plotter.save_page_image(image_page)

        text_regions_p_1 = text_regions_p_1[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        mask_images = (text_regions_p_1[:, :] == 2) * 1
        mask_images = mask_images.astype(np.uint8)
        mask_images = cv2.erode(mask_images[:, :], KERNEL, iterations=10)
        mask_lines = (text_regions_p_1[:, :] == 3) * 1
        mask_lines = mask_lines.astype(np.uint8)
        img_only_regions_with_sep = ((text_regions_p_1[:, :] != 3) & (text_regions_p_1[:, :] != 0)) * 1
        img_only_regions_with_sep = img_only_regions_with_sep.astype(np.uint8)
        
        if erosion_hurts:
            img_only_regions = np.copy(img_only_regions_with_sep[:,:])
        else:
            img_only_regions = cv2.erode(img_only_regions_with_sep[:,:], KERNEL, iterations=6)
            
        try:
            num_col, _ = find_num_col(img_only_regions, num_col_classifier, self.tables, multiplier=6.0)
            num_col = num_col + 1
            if not num_column_is_classified:
                num_col_classifier = num_col + 1
        except Exception as why:
            self.logger.error(why)
            num_col = None
        return num_col, num_col_classifier, img_only_regions, page_coord, image_page, mask_images, mask_lines, text_regions_p_1, cont_page, table_prediction

    def run_enhancement(self,light_version):
        t_in = time.time()
        self.logger.info("Resizing and enhancing image...")
        is_image_enhanced, img_org, img_res, num_col_classifier, num_column_is_classified, img_bin = self.resize_and_enhance_image_with_column_classifier(light_version)
        self.logger.info("Image was %senhanced.", '' if is_image_enhanced else 'not ')
        scale = 1
        if is_image_enhanced:
            if self.allow_enhancement:
                #img_res = img_res.astype(np.uint8)
                self.get_image_and_scales(img_org, img_res, scale)
                if self.plotter:
                    self.plotter.save_enhanced_image(img_res)
            else:
                self.get_image_and_scales_after_enhancing(img_org, img_res)
        else:
            if self.allow_enhancement:
                self.get_image_and_scales(img_org, img_res, scale)
            else:
                self.get_image_and_scales(img_org, img_res, scale)
            if self.allow_scaling:
                img_org, img_res, is_image_enhanced = self.resize_image_with_column_classifier(is_image_enhanced, img_bin)
                self.get_image_and_scales_after_enhancing(img_org, img_res)
        #print("enhancement in ", time.time()-t_in)
        return img_res, is_image_enhanced, num_col_classifier, num_column_is_classified

    def run_textline(self, image_page, num_col_classifier=None):
        scaler_h_textline = 1#1.3  # 1.2#1.2
        scaler_w_textline = 1#1.3  # 0.9#1
        #print(image_page.shape)
        patches = True
        textline_mask_tot_ea, _ = self.textline_contours(image_page, patches, scaler_h_textline, scaler_w_textline, num_col_classifier)
        if self.textline_light:
            textline_mask_tot_ea = textline_mask_tot_ea.astype(np.int16)

        if self.plotter:
            self.plotter.save_plot_of_textlines(textline_mask_tot_ea, image_page)
        return textline_mask_tot_ea

    def run_deskew(self, textline_mask_tot_ea):
        #print(textline_mask_tot_ea.shape, 'textline_mask_tot_ea deskew')
        sigma = 2
        main_page_deskew = True
        n_total_angles = 30
        slope_deskew = return_deskew_slop(cv2.erode(textline_mask_tot_ea, KERNEL, iterations=2), sigma, n_total_angles, main_page_deskew, plotter=self.plotter)
        slope_first = 0

        if self.plotter:
            self.plotter.save_deskewed_image(slope_deskew)
        self.logger.info("slope_deskew: %.2f°", slope_deskew)
        return slope_deskew, slope_first

    def run_marginals(self, image_page, textline_mask_tot_ea, mask_images, mask_lines, num_col_classifier, slope_deskew, text_regions_p_1, table_prediction):
        image_page_rotated, textline_mask_tot = image_page[:, :], textline_mask_tot_ea[:, :]
        textline_mask_tot[mask_images[:, :] == 1] = 0

        text_regions_p_1[mask_lines[:, :] == 1] = 3
        text_regions_p = text_regions_p_1[:, :]
        text_regions_p = np.array(text_regions_p)

        if num_col_classifier in (1, 2):
            try:
                regions_without_separators = (text_regions_p[:, :] == 1) * 1
                if self.tables:
                    regions_without_separators[table_prediction==1] = 1
                regions_without_separators = regions_without_separators.astype(np.uint8)
                text_regions_p = get_marginals(rotate_image(regions_without_separators, slope_deskew), text_regions_p, num_col_classifier, slope_deskew, light_version=self.light_version, kernel=KERNEL)
            except Exception as e:
                self.logger.error("exception %s", e)

        if self.plotter:
            self.plotter.save_plot_of_layout_main_all(text_regions_p, image_page)
            self.plotter.save_plot_of_layout_main(text_regions_p, image_page)
        return textline_mask_tot, text_regions_p, image_page_rotated

    def run_boxes_no_full_layout(self, image_page, textline_mask_tot, text_regions_p, slope_deskew, num_col_classifier, table_prediction, erosion_hurts):
        self.logger.debug('enter run_boxes_no_full_layout')
        t_0_box = time.time()
        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            _, textline_mask_tot_d, text_regions_p_1_n, table_prediction_n = rotation_not_90_func(image_page, textline_mask_tot, text_regions_p, table_prediction, slope_deskew)
            text_regions_p_1_n = resize_image(text_regions_p_1_n, text_regions_p.shape[0], text_regions_p.shape[1])
            textline_mask_tot_d = resize_image(textline_mask_tot_d, text_regions_p.shape[0], text_regions_p.shape[1])
            table_prediction_n = resize_image(table_prediction_n, text_regions_p.shape[0], text_regions_p.shape[1])
            regions_without_separators_d = (text_regions_p_1_n[:, :] == 1) * 1
            if self.tables:
                regions_without_separators_d[table_prediction_n[:,:] == 1] = 1
        regions_without_separators = (text_regions_p[:, :] == 1) * 1  # ( (text_regions_p[:,:]==1) | (text_regions_p[:,:]==2) )*1 #self.return_regions_without_separators_new(text_regions_p[:,:,0],img_only_regions)
        #print(time.time()-t_0_box,'time box in 1')
        if self.tables:
            regions_without_separators[table_prediction ==1 ] = 1
        if np.abs(slope_deskew) < SLOPE_THRESHOLD:
            text_regions_p_1_n = None
            textline_mask_tot_d = None
            regions_without_separators_d = None
        pixel_lines = 3
        if np.abs(slope_deskew) < SLOPE_THRESHOLD:
            _, _, matrix_of_lines_ch, splitter_y_new, _ = find_number_of_columns_in_document(np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2), num_col_classifier, self.tables, pixel_lines)

        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            _, _, matrix_of_lines_ch_d, splitter_y_new_d, _ = find_number_of_columns_in_document(np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2), num_col_classifier, self.tables, pixel_lines)
        #print(time.time()-t_0_box,'time box in 2')
        self.logger.info("num_col_classifier: %s", num_col_classifier)

        if num_col_classifier >= 3:
            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                regions_without_separators = regions_without_separators.astype(np.uint8)
                regions_without_separators = cv2.erode(regions_without_separators[:, :], KERNEL, iterations=6)
            else:
                regions_without_separators_d = regions_without_separators_d.astype(np.uint8)
                regions_without_separators_d = cv2.erode(regions_without_separators_d[:, :], KERNEL, iterations=6)
        #print(time.time()-t_0_box,'time box in 3')
        t1 = time.time()
        if np.abs(slope_deskew) < SLOPE_THRESHOLD:
            boxes, peaks_neg_tot_tables = return_boxes_of_images_by_order_of_reading_new(splitter_y_new, regions_without_separators, matrix_of_lines_ch, num_col_classifier, erosion_hurts, self.tables, self.right2left)
            boxes_d = None
            self.logger.debug("len(boxes): %s", len(boxes))
            #print(time.time()-t_0_box,'time box in 3.1')
            
            if self.tables:
                text_regions_p_tables = np.copy(text_regions_p)
                text_regions_p_tables[:,:][(table_prediction[:,:] == 1)] = 10
                pixel_line = 3
                img_revised_tab2 = self.add_tables_heuristic_to_layout(text_regions_p_tables, boxes, 0, splitter_y_new, peaks_neg_tot_tables, text_regions_p_tables , num_col_classifier , 0.000005, pixel_line)
                #print(time.time()-t_0_box,'time box in 3.2')
                img_revised_tab2, contoures_tables = self.check_iou_of_bounding_box_and_contour_for_tables(img_revised_tab2,table_prediction, 10, num_col_classifier)
                #print(time.time()-t_0_box,'time box in 3.3')
        else:
            boxes_d, peaks_neg_tot_tables_d = return_boxes_of_images_by_order_of_reading_new(splitter_y_new_d, regions_without_separators_d, matrix_of_lines_ch_d, num_col_classifier, erosion_hurts, self.tables, self.right2left)
            boxes = None
            self.logger.debug("len(boxes): %s", len(boxes_d))
            
            if self.tables:
                text_regions_p_tables = np.copy(text_regions_p_1_n)
                text_regions_p_tables =np.round(text_regions_p_tables)
                text_regions_p_tables[:,:][(text_regions_p_tables[:,:] != 3) & (table_prediction_n[:,:] == 1)] = 10
                
                pixel_line = 3
                img_revised_tab2 = self.add_tables_heuristic_to_layout(text_regions_p_tables,boxes_d,0,splitter_y_new_d,peaks_neg_tot_tables_d,text_regions_p_tables, num_col_classifier, 0.000005, pixel_line)
                img_revised_tab2_d,_ = self.check_iou_of_bounding_box_and_contour_for_tables(img_revised_tab2,table_prediction_n, 10, num_col_classifier)
                
                img_revised_tab2_d_rotated = rotate_image(img_revised_tab2_d, -slope_deskew)
                img_revised_tab2_d_rotated = np.round(img_revised_tab2_d_rotated)
                img_revised_tab2_d_rotated = img_revised_tab2_d_rotated.astype(np.int8)
                img_revised_tab2_d_rotated = resize_image(img_revised_tab2_d_rotated, text_regions_p.shape[0], text_regions_p.shape[1])
        #print(time.time()-t_0_box,'time box in 4')
        self.logger.info("detecting boxes took %.1fs", time.time() - t1)
        
        if self.tables:
            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                img_revised_tab = np.copy(img_revised_tab2[:,:,0])
                img_revised_tab[:,:][(text_regions_p[:,:] == 1) & (img_revised_tab[:,:] != 10)] = 1
            else:
                img_revised_tab = np.copy(text_regions_p[:,:])
                img_revised_tab[:,:][img_revised_tab[:,:] == 10] = 0
                img_revised_tab[:,:][img_revised_tab2_d_rotated[:,:,0] == 10] = 10
                
            text_regions_p[:,:][text_regions_p[:,:]==10] = 0
            text_regions_p[:,:][img_revised_tab[:,:]==10] = 10
        else:
            img_revised_tab=text_regions_p[:,:]
        #img_revised_tab = text_regions_p[:, :]
        polygons_of_images = return_contours_of_interested_region(img_revised_tab, 2)

        pixel_img = 4
        min_area_mar = 0.00001
        if self.light_version:
            marginal_mask = (text_regions_p[:,:]==pixel_img)*1
            marginal_mask = marginal_mask.astype('uint8')
            marginal_mask = cv2.dilate(marginal_mask, KERNEL, iterations=2)
            
            polygons_of_marginals = return_contours_of_interested_region(marginal_mask, 1, min_area_mar)
        else:
            polygons_of_marginals = return_contours_of_interested_region(text_regions_p, pixel_img, min_area_mar)
        
        pixel_img = 10
        contours_tables = return_contours_of_interested_region(text_regions_p, pixel_img, min_area_mar)
        #print(time.time()-t_0_box,'time box in 5')
        self.logger.debug('exit run_boxes_no_full_layout')
        return polygons_of_images, img_revised_tab, text_regions_p_1_n, textline_mask_tot_d, regions_without_separators_d, boxes, boxes_d, polygons_of_marginals, contours_tables

    def run_boxes_full_layout(self, image_page, textline_mask_tot, text_regions_p, slope_deskew, num_col_classifier, img_only_regions, table_prediction, erosion_hurts, img_bin_light):
        self.logger.debug('enter run_boxes_full_layout')
        t_full0 = time.time()
        if self.tables:
            if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                image_page_rotated_n,textline_mask_tot_d,text_regions_p_1_n , table_prediction_n = rotation_not_90_func(image_page, textline_mask_tot, text_regions_p, table_prediction, slope_deskew)
                
                text_regions_p_1_n = resize_image(text_regions_p_1_n,text_regions_p.shape[0],text_regions_p.shape[1])
                textline_mask_tot_d = resize_image(textline_mask_tot_d,text_regions_p.shape[0],text_regions_p.shape[1])
                table_prediction_n = resize_image(table_prediction_n,text_regions_p.shape[0],text_regions_p.shape[1])
                
                regions_without_separators_d=(text_regions_p_1_n[:,:] == 1)*1
                regions_without_separators_d[table_prediction_n[:,:] == 1] = 1
            else:
                text_regions_p_1_n = None
                textline_mask_tot_d = None
                regions_without_separators_d = None
                
            regions_without_separators = (text_regions_p[:,:] == 1)*1#( (text_regions_p[:,:]==1) | (text_regions_p[:,:]==2) )*1 #self.return_regions_without_seperators_new(text_regions_p[:,:,0],img_only_regions)
            regions_without_separators[table_prediction == 1] = 1
            
            pixel_lines=3
            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                num_col, peaks_neg_fin, matrix_of_lines_ch, splitter_y_new, seperators_closeup_n = find_number_of_columns_in_document(np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2), num_col_classifier, self.tables, pixel_lines)
            
            if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                num_col_d, peaks_neg_fin_d, matrix_of_lines_ch_d, splitter_y_new_d, seperators_closeup_n_d = find_number_of_columns_in_document(np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2),num_col_classifier, self.tables, pixel_lines)

            if num_col_classifier>=3:
                if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                    regions_without_separators = regions_without_separators.astype(np.uint8)
                    regions_without_separators = cv2.erode(regions_without_separators[:,:], KERNEL, iterations=6)
                
                if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                    regions_without_separators_d = regions_without_separators_d.astype(np.uint8)
                    regions_without_separators_d = cv2.erode(regions_without_separators_d[:,:], KERNEL, iterations=6)
            else:
                pass
            
            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                boxes, peaks_neg_tot_tables = return_boxes_of_images_by_order_of_reading_new(splitter_y_new, regions_without_separators, matrix_of_lines_ch, num_col_classifier, erosion_hurts, self.tables, self.right2left)
                text_regions_p_tables = np.copy(text_regions_p)
                text_regions_p_tables[:,:][(table_prediction[:,:]==1)] = 10
                pixel_line = 3
                img_revised_tab2 = self.add_tables_heuristic_to_layout(text_regions_p_tables, boxes, 0, splitter_y_new, peaks_neg_tot_tables, text_regions_p_tables , num_col_classifier , 0.000005, pixel_line)
                
                img_revised_tab2,contoures_tables = self.check_iou_of_bounding_box_and_contour_for_tables(img_revised_tab2, table_prediction, 10, num_col_classifier)
                
            else:
                boxes_d, peaks_neg_tot_tables_d = return_boxes_of_images_by_order_of_reading_new(splitter_y_new_d, regions_without_separators_d, matrix_of_lines_ch_d, num_col_classifier, erosion_hurts, self.tables, self.right2left)
                text_regions_p_tables = np.copy(text_regions_p_1_n)
                text_regions_p_tables = np.round(text_regions_p_tables)
                text_regions_p_tables[:,:][(text_regions_p_tables[:,:]!=3) & (table_prediction_n[:,:]==1)] = 10
                
                pixel_line = 3
                img_revised_tab2 = self.add_tables_heuristic_to_layout(text_regions_p_tables,boxes_d,0,splitter_y_new_d,peaks_neg_tot_tables_d,text_regions_p_tables, num_col_classifier, 0.000005, pixel_line)
                
                img_revised_tab2_d,_ = self.check_iou_of_bounding_box_and_contour_for_tables(img_revised_tab2, table_prediction_n, 10, num_col_classifier)
                img_revised_tab2_d_rotated = rotate_image(img_revised_tab2_d, -slope_deskew)
                

                img_revised_tab2_d_rotated = np.round(img_revised_tab2_d_rotated)
                img_revised_tab2_d_rotated = img_revised_tab2_d_rotated.astype(np.int8)

                img_revised_tab2_d_rotated = resize_image(img_revised_tab2_d_rotated, text_regions_p.shape[0], text_regions_p.shape[1])


            if np.abs(slope_deskew) < 0.13:
                img_revised_tab = np.copy(img_revised_tab2[:,:,0])
            else:
                img_revised_tab = np.copy(text_regions_p[:,:])
                img_revised_tab[:,:][img_revised_tab[:,:] == 10] = 0
                img_revised_tab[:,:][img_revised_tab2_d_rotated[:,:,0] == 10] = 10
                    
                    
            ##img_revised_tab=img_revised_tab2[:,:,0]
            #img_revised_tab=text_regions_p[:,:]
            text_regions_p[:,:][text_regions_p[:,:]==10] = 0
            text_regions_p[:,:][img_revised_tab[:,:]==10] = 10
            #img_revised_tab[img_revised_tab2[:,:,0]==10] =10
            
        pixel_img = 4
        min_area_mar = 0.00001
        
        if self.light_version:
            marginal_mask = (text_regions_p[:,:]==pixel_img)*1
            marginal_mask = marginal_mask.astype('uint8')
            marginal_mask = cv2.dilate(marginal_mask, KERNEL, iterations=2)
            
            polygons_of_marginals = return_contours_of_interested_region(marginal_mask, 1, min_area_mar)
        else:
            polygons_of_marginals = return_contours_of_interested_region(text_regions_p, pixel_img, min_area_mar)
        
        pixel_img = 10
        contours_tables = return_contours_of_interested_region(text_regions_p, pixel_img, min_area_mar)
        
        # set first model with second model
        text_regions_p[:, :][text_regions_p[:, :] == 2] = 5
        text_regions_p[:, :][text_regions_p[:, :] == 3] = 6
        text_regions_p[:, :][text_regions_p[:, :] == 4] = 8

        image_page = image_page.astype(np.uint8)
        #print("full inside 1", time.time()- t_full0)
        if self.light_version:
            regions_fully, regions_fully_only_drop = self.extract_text_regions_new(img_bin_light, False, cols=num_col_classifier)
        else:
            regions_fully, regions_fully_only_drop = self.extract_text_regions_new(image_page, False, cols=num_col_classifier)
        #print("full inside 2", time.time()- t_full0)
        # 6 is the separators lable in old full layout model
        # 4 is the drop capital class in old full layout model
        # in the new full layout drop capital is 3 and separators are 5
        
        text_regions_p[:,:][regions_fully[:,:,0]==5]=6
        ###regions_fully[:, :, 0][regions_fully_only_drop[:, :, 0] == 3] = 4
        
        #text_regions_p[:,:][regions_fully[:,:,0]==6]=6
        ##regions_fully_only_drop = put_drop_out_from_only_drop_model(regions_fully_only_drop, text_regions_p)
        ##regions_fully[:, :, 0][regions_fully_only_drop[:, :, 0] == 4] = 4
        drop_capital_label_in_full_layout_model = 3
        
        drops = (regions_fully[:,:,0]==drop_capital_label_in_full_layout_model)*1
        
        drops= drops.astype(np.uint8)
        
        regions_fully[:,:,0][regions_fully[:,:,0]==drop_capital_label_in_full_layout_model] = 1
        
        drops = cv2.erode(drops[:,:], KERNEL, iterations=1)
        regions_fully[:,:,0][drops[:,:]==1] = drop_capital_label_in_full_layout_model
        
        
        regions_fully = putt_bb_of_drop_capitals_of_model_in_patches_in_layout(regions_fully, drop_capital_label_in_full_layout_model, text_regions_p)
        ##regions_fully_np, _ = self.extract_text_regions(image_page, False, cols=num_col_classifier)
        ##if num_col_classifier > 2:
            ##regions_fully_np[:, :, 0][regions_fully_np[:, :, 0] == 4] = 0
        ##else:
            ##regions_fully_np = filter_small_drop_capitals_from_no_patch_layout(regions_fully_np, text_regions_p)

        ###regions_fully = boosting_headers_by_longshot_region_segmentation(regions_fully, regions_fully_np, img_only_regions)
        # plt.imshow(regions_fully[:,:,0])
        # plt.show()
        text_regions_p[:, :][regions_fully[:, :, 0] == drop_capital_label_in_full_layout_model] = 4
        ####text_regions_p[:, :][regions_fully_np[:, :, 0] == 4] = 4
        #plt.imshow(text_regions_p)
        #plt.show()
        ####if not self.tables:
        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            _, textline_mask_tot_d, text_regions_p_1_n, regions_fully_n = rotation_not_90_func_full_layout(image_page, textline_mask_tot, text_regions_p, regions_fully, slope_deskew)

            text_regions_p_1_n = resize_image(text_regions_p_1_n, text_regions_p.shape[0], text_regions_p.shape[1])
            textline_mask_tot_d = resize_image(textline_mask_tot_d, text_regions_p.shape[0], text_regions_p.shape[1])
            regions_fully_n = resize_image(regions_fully_n, text_regions_p.shape[0], text_regions_p.shape[1])
            if not self.tables:
                regions_without_separators_d = (text_regions_p_1_n[:, :] == 1) * 1
        else:
            text_regions_p_1_n = None
            textline_mask_tot_d = None
            regions_without_separators_d = None
        if not self.tables:
            regions_without_separators = (text_regions_p[:, :] == 1) * 1
        img_revised_tab = np.copy(text_regions_p[:, :])
        polygons_of_images = return_contours_of_interested_region(img_revised_tab, 5)
        
        self.logger.debug('exit run_boxes_full_layout')
        #print("full inside 3", time.time()- t_full0)
        return polygons_of_images, img_revised_tab, text_regions_p_1_n, textline_mask_tot_d, regions_without_separators_d, regions_fully, regions_without_separators, polygons_of_marginals, contours_tables
    
    def our_load_model(self, model_file):
        
        try:
            model = load_model(model_file, compile=False)
        except:
            model = load_model(model_file , compile=False,custom_objects = {"PatchEncoder": PatchEncoder, "Patches": Patches})

        return model
    def do_order_of_regions_with_machine(self,contours_only_text_parent, contours_only_text_parent_h, text_regions_p):
        y_len = text_regions_p.shape[0]
        x_len = text_regions_p.shape[1]
        
        img_poly = np.zeros((y_len,x_len), dtype='uint8')
        
        unique_pix = np.unique(text_regions_p)

            
        img_poly[text_regions_p[:,:]==1] = 1
        img_poly[text_regions_p[:,:]==2] = 2
        img_poly[text_regions_p[:,:]==3] = 4
        img_poly[text_regions_p[:,:]==6] = 5

        model_ro_machine, _ = self.start_new_session_and_model(self.model_reading_order_machine_dir)

        height1 =672#448
        width1 = 448#224

        height2 =672#448
        width2= 448#224

        height3 =672#448
        width3 = 448#224
        
        img_header_and_sep = np.zeros((y_len,x_len), dtype='uint8')
        
        if contours_only_text_parent_h:
            _, cy_main, x_min_main, x_max_main, y_min_main, y_max_main, _ = find_new_features_of_contours(contours_only_text_parent_h)
            for j in range(len(cy_main)):
                img_header_and_sep[int(y_max_main[j]):int(y_max_main[j])+12,int(x_min_main[j]):int(x_max_main[j]) ] = 1 
            
            co_text_all = contours_only_text_parent + contours_only_text_parent_h
        else:
            co_text_all = contours_only_text_parent


        labels_con = np.zeros((y_len,x_len,len(co_text_all)),dtype='uint8')
        for i in range(len(co_text_all)):
            img_label = np.zeros((y_len,x_len,3),dtype='uint8')
            img_label=cv2.fillPoly(img_label, pts =[co_text_all[i]], color=(1,1,1))
            labels_con[:,:,i] = img_label[:,:,0]
            
            
        img3= np.copy(img_poly)

        labels_con = resize_image(labels_con, height1, width1)

        img_header_and_sep = resize_image(img_header_and_sep, height1, width1)

        img3= resize_image (img3, height3, width3)

        img3 = img3.astype(np.uint16)
        
        
        order_matrix = np.zeros((labels_con.shape[2], labels_con.shape[2]))-1
        inference_bs = 6
        tot_counter = 1
        batch_counter = 0
        i_indexer = []
        j_indexer =[]
        
        input_1= np.zeros( (inference_bs, height1, width1,3))
        
        tot_iteration = int( ( labels_con.shape[2]*(labels_con.shape[2]-1) )/2. )
        full_bs_ite= tot_iteration//inference_bs
        last_bs = tot_iteration % inference_bs
        
        #print(labels_con.shape[2],"number of regions for reading order")
        for i in range(labels_con.shape[2]):
            for j in range(labels_con.shape[2]):
                if j>i:
                    img1= np.repeat(labels_con[:,:,i][:, :, np.newaxis], 3, axis=2)
                    img2 = np.repeat(labels_con[:,:,j][:, :, np.newaxis], 3, axis=2)
                    
                    img2[:,:,0][img3[:,:]==5] = 2
                    img2[:,:,0][img_header_and_sep[:,:]==1] = 3
                    
                    img1[:,:,0][img3[:,:]==5] = 2
                    img1[:,:,0][img_header_and_sep[:,:]==1] = 3
                    
                    
                    i_indexer.append(i)
                    j_indexer.append(j)
                    
                    input_1[batch_counter,:,:,0] = img1[:,:,0]/3.
                    input_1[batch_counter,:,:,2] = img2[:,:,0]/3.
                    input_1[batch_counter,:,:,1] = img3[:,:]/5.
                    
                    batch_counter = batch_counter+1
                    
                    if batch_counter==inference_bs or ( (tot_counter//inference_bs)==full_bs_ite and tot_counter%inference_bs==last_bs):
                        y_pr=model_ro_machine.predict(input_1 , verbose=0)

                        if batch_counter==inference_bs:
                            iteration_batches = inference_bs
                        else:
                            iteration_batches = last_bs
                        for jb in range(iteration_batches):
                            if y_pr[jb][0]>=0.5:
                                order_class = 1
                            else:
                                order_class = 0
                                
                            order_matrix[i_indexer[jb],j_indexer[jb]] = y_pr[jb][0]#order_class
                            order_matrix[j_indexer[jb],i_indexer[jb]] = 1-y_pr[jb][0]#int( 1 - order_class)
                        
                        batch_counter = 0
                        
                        i_indexer = []
                        j_indexer = []
                    tot_counter = tot_counter+1
                    
                    
        sum_mat = np.sum(order_matrix, axis=1)
        index_sort = np.argsort(sum_mat)
        index_sort = index_sort[::-1]
        
        REGION_ID_TEMPLATE = 'region_%04d'
        order_of_texts = []
        id_of_texts = []
        for order, id_text in enumerate(index_sort):
            order_of_texts.append(id_text)
            id_of_texts.append( REGION_ID_TEMPLATE % order )
            
        
        return order_of_texts, id_of_texts
    
    def update_list_and_return_first_with_length_bigger_than_one(self,index_element_to_be_updated, innner_index_pr_pos, pr_list, pos_list,list_inp):
        list_inp.pop(index_element_to_be_updated)
        if len(pr_list)>0:
            list_inp.insert(index_element_to_be_updated, pr_list)
        else:
            index_element_to_be_updated = index_element_to_be_updated -1
        
        list_inp.insert(index_element_to_be_updated+1, [innner_index_pr_pos])
        if len(pos_list)>0:
            list_inp.insert(index_element_to_be_updated+2, pos_list)
        
        len_all_elements = [len(i) for i in list_inp]
        list_len_bigger_1 = np.where(np.array(len_all_elements)>1)
        list_len_bigger_1 = list_len_bigger_1[0]
        
        if len(list_len_bigger_1)>0:
            early_list_bigger_than_one = list_len_bigger_1[0]
        else:
            early_list_bigger_than_one = -20
        return list_inp, early_list_bigger_than_one
    def do_order_of_regions_with_machine_optimized_algorithm(self,contours_only_text_parent, contours_only_text_parent_h, text_regions_p):
        y_len = text_regions_p.shape[0]
        x_len = text_regions_p.shape[1]
        
        img_poly = np.zeros((y_len,x_len), dtype='uint8')
        
        unique_pix = np.unique(text_regions_p)

            
        img_poly[text_regions_p[:,:]==1] = 1
        img_poly[text_regions_p[:,:]==2] = 2
        img_poly[text_regions_p[:,:]==3] = 4
        img_poly[text_regions_p[:,:]==6] = 5
            
        if self.dir_in:
            pass
        else:
            self.model_reading_order_machine, _ = self.start_new_session_and_model(self.model_reading_order_machine_dir)

        height1 =672#448
        width1 = 448#224

        height2 =672#448
        width2= 448#224

        height3 =672#448
        width3 = 448#224
        
        img_header_and_sep = np.zeros((y_len,x_len), dtype='uint8')
        
        if contours_only_text_parent_h:
            _, cy_main, x_min_main, x_max_main, y_min_main, y_max_main, _ = find_new_features_of_contours(contours_only_text_parent_h)
            
            for j in range(len(cy_main)):
                img_header_and_sep[int(y_max_main[j]):int(y_max_main[j])+12,int(x_min_main[j]):int(x_max_main[j]) ] = 1 
            
            co_text_all = contours_only_text_parent + contours_only_text_parent_h
        else:
            co_text_all = contours_only_text_parent


        labels_con = np.zeros((y_len,x_len,len(co_text_all)),dtype='uint8')
        for i in range(len(co_text_all)):
            img_label = np.zeros((y_len,x_len,3),dtype='uint8')
            img_label=cv2.fillPoly(img_label, pts =[co_text_all[i]], color=(1,1,1))
            labels_con[:,:,i] = img_label[:,:,0]
            
            
        img3= np.copy(img_poly)

        labels_con = resize_image(labels_con, height1, width1)

        img_header_and_sep = resize_image(img_header_and_sep, height1, width1)

        img3= resize_image (img3, height3, width3)

        img3 = img3.astype(np.uint16)
        
        inference_bs = 3
        input_1= np.zeros( (inference_bs, height1, width1,3))
        starting_list_of_regions = []
        starting_list_of_regions.append( list(range(labels_con.shape[2])) )
        index_update = 0
        index_selected = starting_list_of_regions[0]
        #print(labels_con.shape[2],"number of regions for reading order")
        while index_update>=0:
            ij_list = starting_list_of_regions[index_update] 
            i = ij_list[0]
            ij_list.pop(0)
            
            pr_list = []
            post_list = []
            
            batch_counter = 0
            tot_counter = 1
            
            tot_iteration = len(ij_list)
            full_bs_ite= tot_iteration//inference_bs
            last_bs = tot_iteration % inference_bs
            
            jbatch_indexer =[]
            for j in ij_list:
                img1= np.repeat(labels_con[:,:,i][:, :, np.newaxis], 3, axis=2)
                img2 = np.repeat(labels_con[:,:,j][:, :, np.newaxis], 3, axis=2)
                
                img2[:,:,0][img3[:,:]==5] = 2
                img2[:,:,0][img_header_and_sep[:,:]==1] = 3
                
                img1[:,:,0][img3[:,:]==5] = 2
                img1[:,:,0][img_header_and_sep[:,:]==1] = 3

                jbatch_indexer.append(j)
                    
                input_1[batch_counter,:,:,0] = img1[:,:,0]/3.
                input_1[batch_counter,:,:,2] = img2[:,:,0]/3.
                input_1[batch_counter,:,:,1] = img3[:,:]/5.

                batch_counter = batch_counter+1
                
                if batch_counter==inference_bs or ( (tot_counter//inference_bs)==full_bs_ite and tot_counter%inference_bs==last_bs):
                    y_pr=self.model_reading_order_machine.predict(input_1 , verbose=0)
                    
                    if batch_counter==inference_bs:
                        iteration_batches = inference_bs
                    else:
                        iteration_batches = last_bs
                    for jb in range(iteration_batches):
                        if y_pr[jb][0]>=0.5:
                            post_list.append(jbatch_indexer[jb])
                        else:
                            pr_list.append(jbatch_indexer[jb])
                            
                    batch_counter = 0
                    jbatch_indexer = []
                    
                tot_counter = tot_counter+1
                    
            starting_list_of_regions, index_update = self.update_list_and_return_first_with_length_bigger_than_one(index_update, i, pr_list, post_list,starting_list_of_regions)

        index_sort = [i[0] for i in starting_list_of_regions ]
        
        REGION_ID_TEMPLATE = 'region_%04d'
        order_of_texts = []
        id_of_texts = []
        for order, id_text in enumerate(index_sort):
            order_of_texts.append(id_text)
            id_of_texts.append( REGION_ID_TEMPLATE % order )
            
        
        return order_of_texts, id_of_texts
    def return_start_and_end_of_common_text_of_textline_ocr(self,textline_image, ind_tot):
        width = np.shape(textline_image)[1]
        height = np.shape(textline_image)[0]
        common_window = int(0.2*width)

        width1 = int ( width/2. - common_window )
        width2 = int ( width/2. + common_window )
        
        img_sum = np.sum(textline_image[:,:,0], axis=0)
        sum_smoothed = gaussian_filter1d(img_sum, 3)
        
        peaks_real, _ = find_peaks(sum_smoothed, height=0)
        
        if len(peaks_real)>70:
            print(len(peaks_real), 'len(peaks_real)')

            peaks_real = peaks_real[(peaks_real<width2) & (peaks_real>width1)]

            arg_sort = np.argsort(sum_smoothed[peaks_real])

            arg_sort4 =arg_sort[::-1][:4]

            peaks_sort_4 = peaks_real[arg_sort][::-1][:4]

            argsort_sorted = np.argsort(peaks_sort_4)

            first_4_sorted = peaks_sort_4[argsort_sorted]
            y_4_sorted = sum_smoothed[peaks_real][arg_sort4[argsort_sorted]]
            #print(first_4_sorted,'first_4_sorted')
            
            arg_sortnew = np.argsort(y_4_sorted)
            peaks_final =np.sort( first_4_sorted[arg_sortnew][2:] )
            
            #plt.figure(ind_tot)
            #plt.imshow(textline_image)
            #plt.plot([peaks_final[0], peaks_final[0]], [0, height-1])
            #plt.plot([peaks_final[1], peaks_final[1]], [0, height-1])
            #plt.savefig('./'+str(ind_tot)+'.png')
            
            return peaks_final[0], peaks_final[1]
        else:
            pass
        
        
    def return_start_and_end_of_common_text_of_textline_ocr_without_common_section(self,textline_image, ind_tot):
        width = np.shape(textline_image)[1]
        height = np.shape(textline_image)[0]
        common_window = int(0.06*width)

        width1 = int ( width/2. - common_window )
        width2 = int ( width/2. + common_window )
        
        img_sum = np.sum(textline_image[:,:,0], axis=0)
        sum_smoothed = gaussian_filter1d(img_sum, 3)
        
        peaks_real, _ = find_peaks(sum_smoothed, height=0)
        
        if len(peaks_real)>70:
            #print(len(peaks_real), 'len(peaks_real)')

            peaks_real = peaks_real[(peaks_real<width2) & (peaks_real>width1)]

            arg_max = np.argmax(sum_smoothed[peaks_real])

            peaks_final = peaks_real[arg_max]
            
            #plt.figure(ind_tot)
            #plt.imshow(textline_image)
            #plt.plot([peaks_final, peaks_final], [0, height-1])
            ##plt.plot([peaks_final[1], peaks_final[1]], [0, height-1])
            #plt.savefig('./'+str(ind_tot)+'.png')
            
            return peaks_final
        else:
            return None
    def return_start_and_end_of_common_text_of_textline_ocr_new_splitted(self,peaks_real, sum_smoothed, start_split, end_split):
        peaks_real = peaks_real[(peaks_real<end_split) & (peaks_real>start_split)]

        arg_sort = np.argsort(sum_smoothed[peaks_real])

        arg_sort4 =arg_sort[::-1][:4]

        peaks_sort_4 = peaks_real[arg_sort][::-1][:4]

        argsort_sorted = np.argsort(peaks_sort_4)

        first_4_sorted = peaks_sort_4[argsort_sorted]
        y_4_sorted = sum_smoothed[peaks_real][arg_sort4[argsort_sorted]]
        #print(first_4_sorted,'first_4_sorted')
        
        arg_sortnew = np.argsort(y_4_sorted)
        peaks_final =np.sort( first_4_sorted[arg_sortnew][3:] )
        return peaks_final[0]
        
    def return_start_and_end_of_common_text_of_textline_ocr_new(self,textline_image, ind_tot):
        width = np.shape(textline_image)[1]
        height = np.shape(textline_image)[0]
        common_window = int(0.15*width)

        width1 = int ( width/2. - common_window )
        width2 = int ( width/2. + common_window )
        mid = int(width/2.)
        
        img_sum = np.sum(textline_image[:,:,0], axis=0)
        sum_smoothed = gaussian_filter1d(img_sum, 3)
        
        peaks_real, _ = find_peaks(sum_smoothed, height=0)
        
        if len(peaks_real)>70:
            peak_start = self.return_start_and_end_of_common_text_of_textline_ocr_new_splitted(peaks_real, sum_smoothed, width1, mid+2)

            peak_end = self.return_start_and_end_of_common_text_of_textline_ocr_new_splitted(peaks_real, sum_smoothed, mid-2, width2)
            
            #plt.figure(ind_tot)
            #plt.imshow(textline_image)
            #plt.plot([peak_start, peak_start], [0, height-1])
            #plt.plot([peak_end, peak_end], [0, height-1])
            #plt.savefig('./'+str(ind_tot)+'.png')
            
            return peak_start, peak_end
        else:
            pass
    
    def return_ocr_of_textline_without_common_section(self, textline_image, model_ocr, processor, device, width_textline, h2w_ratio,ind_tot):
        if h2w_ratio > 0.05:
            pixel_values = processor(textline_image, return_tensors="pt").pixel_values
            generated_ids = model_ocr.generate(pixel_values.to(device))
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            
            #width = np.shape(textline_image)[1]
            #height = np.shape(textline_image)[0]
            #common_window = int(0.3*width)
            
            #width1 = int ( width/2. - common_window )
            #width2 = int ( width/2. + common_window )
            
            
            split_point = self.return_start_and_end_of_common_text_of_textline_ocr_without_common_section(textline_image, ind_tot)
            if split_point:
                image1 = textline_image[:, :split_point,:]# image.crop((0, 0, width2, height))
                image2 = textline_image[:, split_point:,:]#image.crop((width1, 0, width, height))
                
                #pixel_values1 = processor(image1, return_tensors="pt").pixel_values
                #pixel_values2 = processor(image2, return_tensors="pt").pixel_values
                
                pixel_values_merged = processor([image1,image2], return_tensors="pt").pixel_values
                generated_ids_merged = model_ocr.generate(pixel_values_merged.to(device))
                generated_text_merged = processor.batch_decode(generated_ids_merged, skip_special_tokens=True)
                
                #print(generated_text_merged,'generated_text_merged')
                
                #generated_ids1 = model_ocr.generate(pixel_values1.to(device))
                #generated_ids2 = model_ocr.generate(pixel_values2.to(device))
                
                #generated_text1 = processor.batch_decode(generated_ids1, skip_special_tokens=True)[0]
                #generated_text2 = processor.batch_decode(generated_ids2, skip_special_tokens=True)[0]
                
                #generated_text = generated_text1 + ' ' + generated_text2
                generated_text = generated_text_merged[0] + ' ' + generated_text_merged[1]
            
                #print(generated_text1,'generated_text1')
                #print(generated_text2, 'generated_text2')
                #print('########################################')
            else:
                pixel_values = processor(textline_image, return_tensors="pt").pixel_values
                generated_ids = model_ocr.generate(pixel_values.to(device))
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
          
        #print(generated_text,'generated_text')
        #print('########################################')
        return generated_text
    def return_ocr_of_textline(self, textline_image, model_ocr, processor, device, width_textline, h2w_ratio,ind_tot):
        if h2w_ratio > 0.05:
            pixel_values = processor(textline_image, return_tensors="pt").pixel_values
            generated_ids = model_ocr.generate(pixel_values.to(device))
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            #width = np.shape(textline_image)[1]
            #height = np.shape(textline_image)[0]
            #common_window = int(0.3*width)
            
            #width1 = int ( width/2. - common_window )
            #width2 = int ( width/2. + common_window )
            
            try:
                width1, width2 = self.return_start_and_end_of_common_text_of_textline_ocr_new(textline_image, ind_tot)
            
                image1 = textline_image[:, :width2,:]# image.crop((0, 0, width2, height))
                image2 = textline_image[:, width1:,:]#image.crop((width1, 0, width, height))
                
                pixel_values1 = processor(image1, return_tensors="pt").pixel_values
                pixel_values2 = processor(image2, return_tensors="pt").pixel_values
                
                generated_ids1 = model_ocr.generate(pixel_values1.to(device))
                generated_ids2 = model_ocr.generate(pixel_values2.to(device))
                
                generated_text1 = processor.batch_decode(generated_ids1, skip_special_tokens=True)[0]
                generated_text2 = processor.batch_decode(generated_ids2, skip_special_tokens=True)[0]
                #print(generated_text1,'generated_text1')
                #print(generated_text2, 'generated_text2')
                #print('########################################')
            
                match = sq(None, generated_text1, generated_text2).find_longest_match(0, len(generated_text1), 0, len(generated_text2))
                
                generated_text = generated_text1 + generated_text2[match.b+match.size:]
            except:
                pixel_values = processor(textline_image, return_tensors="pt").pixel_values
                generated_ids = model_ocr.generate(pixel_values.to(device))
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
        return generated_text
    
    def return_textline_contour_with_added_box_coordinate(self, textline_contour,  box_ind):
        textline_contour[:,0] = textline_contour[:,0] + box_ind[2]
        textline_contour[:,1] = textline_contour[:,1] + box_ind[0]
        return textline_contour
    def return_list_of_contours_with_desired_order(self, ls_cons, sorted_indexes):
        return [ls_cons[sorted_indexes[index]] for index in range(len(sorted_indexes))]
    
    def return_it_in_two_groups(self,x_differential):
        split = [ind if x_differential[ind]!=x_differential[ind+1] else -1 for ind in range(len(x_differential)-1)]

        split_masked = list( np.array(split[:])[np.array(split[:])!=-1] )

        if 0 not in split_masked:
            split_masked.insert(0, -1)

        split_masked.append(len(x_differential)-1)

        split_masked = np.array(split_masked) +1

        sums = [np.sum(x_differential[split_masked[ind]:split_masked[ind+1]]) for ind in range(len(split_masked)-1)]

        indexes_to_bec_changed = [ind if ( np.abs(sums[ind-1]) > np.abs(sums[ind]) and  np.abs(sums[ind+1]) > np.abs(sums[ind])) else -1 for ind in range(1,len(sums)-1)  ]

        indexes_to_bec_changed_filtered = np.array(indexes_to_bec_changed)[np.array(indexes_to_bec_changed)!=-1]

        x_differential_new = np.copy(x_differential)
        for i in indexes_to_bec_changed_filtered:
            x_differential_new[split_masked[i]:split_masked[i+1]] = -1*np.array(x_differential)[split_masked[i]:split_masked[i+1]]
            
        return x_differential_new
    def dilate_textregions_contours_textline_version(self,all_found_textline_polygons):
        #print(all_found_textline_polygons)
        
        for j in range(len(all_found_textline_polygons)):
            for ij in range(len(all_found_textline_polygons[j])):
                
                con_ind = all_found_textline_polygons[j][ij]
                area = cv2.contourArea(con_ind)
                con_ind = con_ind.astype(np.float)
                
                x_differential = np.diff( con_ind[:,0,0])
                y_differential = np.diff( con_ind[:,0,1])
                
                
                x_differential = gaussian_filter1d(x_differential, 0.1)
                y_differential = gaussian_filter1d(y_differential, 0.1)
                
                x_min = float(np.min( con_ind[:,0,0] ))
                y_min = float(np.min( con_ind[:,0,1] ))
                
                x_max = float(np.max( con_ind[:,0,0] ))
                y_max = float(np.max( con_ind[:,0,1] ))
                
                x_differential_mask_nonzeros = [ ind/abs(ind) if ind!=0 else ind for ind in x_differential]
                y_differential_mask_nonzeros = [ ind/abs(ind) if ind!=0 else ind for ind in y_differential]
                
                abs_diff=abs(abs(x_differential)- abs(y_differential) )
                
                inc_x = np.zeros(len(x_differential)+1)
                inc_y = np.zeros(len(x_differential)+1)
                
                
                if (y_max-y_min) <= (x_max-x_min):
                    dilation_m1 = round(area / (x_max-x_min) * 0.12)
                else:
                    dilation_m1 = round(area / (y_max-y_min) * 0.12)
                    
                if dilation_m1>8:
                    dilation_m1 = 8
                if dilation_m1<6:
                    dilation_m1 = 6
                #print(dilation_m1, 'dilation_m1')
                dilation_m1 = 6
                dilation_m2 = int(dilation_m1/2.) +1 
            
                for i in range(len(x_differential)):
                    if abs_diff[i]==0:
                        inc_x[i+1] = dilation_m2*(-1*y_differential_mask_nonzeros[i])
                        inc_y[i+1] = dilation_m2*(x_differential_mask_nonzeros[i])
                    elif abs_diff[i]!=0 and x_differential_mask_nonzeros[i]==0 and y_differential_mask_nonzeros[i]!=0:
                        inc_x[i+1]= dilation_m1*(-1*y_differential_mask_nonzeros[i])
                    elif abs_diff[i]!=0 and x_differential_mask_nonzeros[i]!=0 and y_differential_mask_nonzeros[i]==0:
                        inc_y[i+1] = dilation_m1*(x_differential_mask_nonzeros[i])
                    
                    elif abs_diff[i]!=0 and abs_diff[i]>=3:
                        if abs(x_differential[i])>abs(y_differential[i]):
                            inc_y[i+1] = dilation_m1*(x_differential_mask_nonzeros[i])
                        else:
                            inc_x[i+1]= dilation_m1*(-1*y_differential_mask_nonzeros[i])
                    else:
                        inc_x[i+1] = dilation_m2*(-1*y_differential_mask_nonzeros[i])
                        inc_y[i+1] = dilation_m2*(x_differential_mask_nonzeros[i])
                
                
                inc_x[0] = inc_x[-1]
                inc_y[0] = inc_y[-1]
                
                con_scaled = con_ind*1
                
                con_scaled[:,0, 0] = con_ind[:,0,0] + np.array(inc_x)[:]
                con_scaled[:,0, 1] = con_ind[:,0,1] + np.array(inc_y)[:]
                
                con_scaled[:,0, 1][con_scaled[:,0, 1]<0] = 0
                con_scaled[:,0, 0][con_scaled[:,0, 0]<0] = 0
                
                area_scaled = cv2.contourArea(con_scaled.astype(np.int32))
                
                con_ind = con_ind.astype(np.int32)
                
                results = [cv2.pointPolygonTest(con_ind, (con_scaled[ind,0, 0], con_scaled[ind,0, 1]), False) for ind in range(len(con_scaled[:,0, 1])) ]
                
                results = np.array(results)
                
                #print(results,'results')
                
                results[results==0] = 1
                
                
                diff_result = np.diff(results)
                
                indices_2 = [ind for ind in range(len(diff_result)) if diff_result[ind]==2]
                indices_m2 = [ind for ind in range(len(diff_result)) if diff_result[ind]==-2]

                    
                if results[0]==1:
                    con_scaled[:indices_m2[0]+1,0, 1] = con_ind[:indices_m2[0]+1,0,1]
                    con_scaled[:indices_m2[0]+1,0, 0] = con_ind[:indices_m2[0]+1,0,0]
                    #indices_2 = indices_2[1:]
                    indices_m2 = indices_m2[1:]
                    
                    
                    
                if len(indices_2)>len(indices_m2):
                    con_scaled[indices_2[-1]+1:,0, 1] = con_ind[indices_2[-1]+1:,0,1]
                    con_scaled[indices_2[-1]+1:,0, 0] = con_ind[indices_2[-1]+1:,0,0]
                    
                    indices_2 = indices_2[:-1]
                    
                
                for ii in range(len(indices_2)):
                    con_scaled[indices_2[ii]+1:indices_m2[ii]+1,0, 1] = con_scaled[indices_2[ii],0, 1]
                    con_scaled[indices_2[ii]+1:indices_m2[ii]+1,0, 0] = con_scaled[indices_2[ii],0, 0]
                    

                all_found_textline_polygons[j][ij][:,0,1] = con_scaled[:,0, 1]
                all_found_textline_polygons[j][ij][:,0,0] = con_scaled[:,0, 0]
        return all_found_textline_polygons
    def dilate_textregions_contours(self,all_found_textline_polygons):
        #print(all_found_textline_polygons)
        for j in range(len(all_found_textline_polygons)):
            
            con_ind = all_found_textline_polygons[j]
            #print(len(con_ind[:,0,0]),'con_ind[:,0,0]')
            area = cv2.contourArea(con_ind)
            con_ind = con_ind.astype(np.float)
            
            x_differential = np.diff( con_ind[:,0,0])
            y_differential = np.diff( con_ind[:,0,1])
            
            
            x_differential = gaussian_filter1d(x_differential, 0.1)
            y_differential = gaussian_filter1d(y_differential, 0.1)
            
            x_min = float(np.min( con_ind[:,0,0] ))
            y_min = float(np.min( con_ind[:,0,1] ))
            
            x_max = float(np.max( con_ind[:,0,0] ))
            y_max = float(np.max( con_ind[:,0,1] ))
            
            x_differential_mask_nonzeros = [ ind/abs(ind) if ind!=0 else ind for ind in x_differential]
            y_differential_mask_nonzeros = [ ind/abs(ind) if ind!=0 else ind for ind in y_differential]
            
            abs_diff=abs(abs(x_differential)- abs(y_differential) )
            
            inc_x = np.zeros(len(x_differential)+1)
            inc_y = np.zeros(len(x_differential)+1)
            
            
            if (y_max-y_min) <= (x_max-x_min):
                dilation_m1 = round(area / (x_max-x_min) * 0.12)
            else:
                dilation_m1 = round(area / (y_max-y_min) * 0.12)
                
            if dilation_m1>8:
                dilation_m1 = 8
            if dilation_m1<6:
                dilation_m1 = 6
            #print(dilation_m1, 'dilation_m1')
            dilation_m1 = 6
            dilation_m2 = int(dilation_m1/2.) +1 
            
            for i in range(len(x_differential)):
                if abs_diff[i]==0:
                    inc_x[i+1] = dilation_m2*(-1*y_differential_mask_nonzeros[i])
                    inc_y[i+1] = dilation_m2*(x_differential_mask_nonzeros[i])
                elif abs_diff[i]!=0 and x_differential_mask_nonzeros[i]==0 and y_differential_mask_nonzeros[i]!=0:
                    inc_x[i+1]= dilation_m1*(-1*y_differential_mask_nonzeros[i])
                elif abs_diff[i]!=0 and x_differential_mask_nonzeros[i]!=0 and y_differential_mask_nonzeros[i]==0:
                    inc_y[i+1] = dilation_m1*(x_differential_mask_nonzeros[i])
                
                elif abs_diff[i]!=0 and abs_diff[i]>=3:
                    if abs(x_differential[i])>abs(y_differential[i]):
                        inc_y[i+1] = dilation_m1*(x_differential_mask_nonzeros[i])
                    else:
                        inc_x[i+1]= dilation_m1*(-1*y_differential_mask_nonzeros[i])
                else:
                    inc_x[i+1] = dilation_m2*(-1*y_differential_mask_nonzeros[i])
                    inc_y[i+1] = dilation_m2*(x_differential_mask_nonzeros[i])
            
            
            inc_x[0] = inc_x[-1]
            inc_y[0] = inc_y[-1]
            
            con_scaled = con_ind*1
            
            con_scaled[:,0, 0] = con_ind[:,0,0] + np.array(inc_x)[:]
            con_scaled[:,0, 1] = con_ind[:,0,1] + np.array(inc_y)[:]
            
            con_scaled[:,0, 1][con_scaled[:,0, 1]<0] = 0
            con_scaled[:,0, 0][con_scaled[:,0, 0]<0] = 0
            
            area_scaled = cv2.contourArea(con_scaled.astype(np.int32))
            
            con_ind = con_ind.astype(np.int32)
            
            results = [cv2.pointPolygonTest(con_ind, (con_scaled[ind,0, 0], con_scaled[ind,0, 1]), False) for ind in range(len(con_scaled[:,0, 1])) ]
            
            results = np.array(results)
            
            #print(results,'results')
            
            results[results==0] = 1
            
            
            diff_result = np.diff(results)
            
            indices_2 = [ind for ind in range(len(diff_result)) if diff_result[ind]==2]
            indices_m2 = [ind for ind in range(len(diff_result)) if diff_result[ind]==-2]

                
            if results[0]==1:
                con_scaled[:indices_m2[0]+1,0, 1] = con_ind[:indices_m2[0]+1,0,1]
                con_scaled[:indices_m2[0]+1,0, 0] = con_ind[:indices_m2[0]+1,0,0]
                #indices_2 = indices_2[1:]
                indices_m2 = indices_m2[1:]
                
                
                
            if len(indices_2)>len(indices_m2):
                con_scaled[indices_2[-1]+1:,0, 1] = con_ind[indices_2[-1]+1:,0,1]
                con_scaled[indices_2[-1]+1:,0, 0] = con_ind[indices_2[-1]+1:,0,0]
                
                indices_2 = indices_2[:-1]
                
            
            for ii in range(len(indices_2)):
                con_scaled[indices_2[ii]+1:indices_m2[ii]+1,0, 1] = con_scaled[indices_2[ii],0, 1]
                con_scaled[indices_2[ii]+1:indices_m2[ii]+1,0, 0] = con_scaled[indices_2[ii],0, 0]
                

            all_found_textline_polygons[j][:,0,1] = con_scaled[:,0, 1]
            all_found_textline_polygons[j][:,0,0] = con_scaled[:,0, 0]
        return all_found_textline_polygons
                    
            
    def dilate_textline_contours(self,all_found_textline_polygons):
        for j in range(len(all_found_textline_polygons)):
            for ij in range(len(all_found_textline_polygons[j])):
            
                con_ind = all_found_textline_polygons[j][ij]
                area = cv2.contourArea(con_ind)
                
                con_ind = con_ind.astype(np.float)
                
                x_differential = np.diff( con_ind[:,0,0])
                y_differential = np.diff( con_ind[:,0,1])
                
                x_differential = gaussian_filter1d(x_differential, 3)
                y_differential = gaussian_filter1d(y_differential, 3)
                
                x_min = float(np.min( con_ind[:,0,0] ))
                y_min = float(np.min( con_ind[:,0,1] ))
                
                x_max = float(np.max( con_ind[:,0,0] ))
                y_max = float(np.max( con_ind[:,0,1] ))
                
                x_differential_mask_nonzeros = [ ind/abs(ind) if ind!=0 else ind for ind in x_differential]
                y_differential_mask_nonzeros = [ ind/abs(ind) if ind!=0 else ind for ind in y_differential]
                
                abs_diff=abs(abs(x_differential)- abs(y_differential) )
                
                inc_x = np.zeros(len(x_differential)+1)
                inc_y = np.zeros(len(x_differential)+1)
                    
                if (y_max-y_min) <= (x_max-x_min):
                    dilation_m1 = round(area / (x_max-x_min) * 0.35)
                else:
                    dilation_m1 = round(area / (y_max-y_min) * 0.35)
                    
                  
                if dilation_m1>12:
                    dilation_m1 = 12
                if dilation_m1<4:
                    dilation_m1 = 4
                #print(dilation_m1, 'dilation_m1')
                dilation_m2 = int(dilation_m1/2.) +1
                
                for i in range(len(x_differential)):
                    if abs_diff[i]==0:
                        inc_x[i+1] = dilation_m2*(-1*y_differential_mask_nonzeros[i])
                        inc_y[i+1] = dilation_m2*(x_differential_mask_nonzeros[i])
                    elif abs_diff[i]!=0 and x_differential_mask_nonzeros[i]==0 and y_differential_mask_nonzeros[i]!=0:
                        inc_x[i+1]= dilation_m1*(-1*y_differential_mask_nonzeros[i])
                    elif abs_diff[i]!=0 and x_differential_mask_nonzeros[i]!=0 and y_differential_mask_nonzeros[i]==0:
                        inc_y[i+1] = dilation_m1*(x_differential_mask_nonzeros[i])
                    
                    elif abs_diff[i]!=0 and abs_diff[i]>=3:
                        if abs(x_differential[i])>abs(y_differential[i]):
                            inc_y[i+1] = dilation_m1*(x_differential_mask_nonzeros[i])
                        else:
                            inc_x[i+1]= dilation_m1*(-1*y_differential_mask_nonzeros[i])
                    else:
                        inc_x[i+1] = dilation_m2*(-1*y_differential_mask_nonzeros[i])
                        inc_y[i+1] = dilation_m2*(x_differential_mask_nonzeros[i])
                        
                
                inc_x[0] = inc_x[-1]
                inc_y[0] = inc_y[-1]
                
                con_scaled = con_ind*1
                
                con_scaled[:,0, 0] = con_ind[:,0,0] + np.array(inc_x)[:]
                con_scaled[:,0, 1] = con_ind[:,0,1] + np.array(inc_y)[:]
                
                con_scaled[:,0, 1][con_scaled[:,0, 1]<0] = 0
                con_scaled[:,0, 0][con_scaled[:,0, 0]<0] = 0
                
                
                con_ind = con_ind.astype(np.int32)
                
                results = [cv2.pointPolygonTest(con_ind, (con_scaled[ind,0, 0], con_scaled[ind,0, 1]), False) for ind in range(len(con_scaled[:,0, 1])) ]
                
                results = np.array(results)
                
                results[results==0] = 1
                
                
                diff_result = np.diff(results)
                
                indices_2 = [ind for ind in range(len(diff_result)) if diff_result[ind]==2]
                indices_m2 = [ind for ind in range(len(diff_result)) if diff_result[ind]==-2]
                    
                if results[0]==1:
                    con_scaled[:indices_m2[0]+1,0, 1] = con_ind[:indices_m2[0]+1,0,1]
                    con_scaled[:indices_m2[0]+1,0, 0] = con_ind[:indices_m2[0]+1,0,0]
                    indices_m2 = indices_m2[1:]
                    
                    
                    
                if len(indices_2)>len(indices_m2):
                    con_scaled[indices_2[-1]+1:,0, 1] = con_ind[indices_2[-1]+1:,0,1]
                    con_scaled[indices_2[-1]+1:,0, 0] = con_ind[indices_2[-1]+1:,0,0]
                    indices_2 = indices_2[:-1]
                    
                
                for ii in range(len(indices_2)):
                    con_scaled[indices_2[ii]+1:indices_m2[ii]+1,0, 1] = con_scaled[indices_2[ii],0, 1]
                    con_scaled[indices_2[ii]+1:indices_m2[ii]+1,0, 0] = con_scaled[indices_2[ii],0, 0]
                
                all_found_textline_polygons[j][ij][:,0,1] = con_scaled[:,0, 1]
                all_found_textline_polygons[j][ij][:,0,0] = con_scaled[:,0, 0]
        return all_found_textline_polygons
    
    def filter_contours_inside_a_bigger_one(self,contours, image, marginal_cnts=None, type_contour="textregion"):
        if type_contour=="textregion":
            areas = [cv2.contourArea(contours[j]) for j in range(len(contours))]
            area_tot = image.shape[0]*image.shape[1]
            
            M_main = [cv2.moments(contours[j]) for j in range(len(contours))]
            cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
            cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
            

            
            areas_ratio = np.array(areas)/ area_tot
            contours_index_small = [ind for ind in range(len(contours)) if areas_ratio[ind] < 1e-3]
            contours_index_big = [ind  for ind in range(len(contours)) if areas_ratio[ind] >= 1e-3]
            
            #contours_> = [contours[ind] for ind in contours_index_big]
            indexes_to_be_removed = []
            for ind_small in contours_index_small:
                results = [cv2.pointPolygonTest(contours[ind], (cx_main[ind_small], cy_main[ind_small]), False) for ind in contours_index_big ]
                if marginal_cnts:
                    results_marginal = [cv2.pointPolygonTest(marginal_cnts[ind], (cx_main[ind_small], cy_main[ind_small]), False) for ind in range(len(marginal_cnts)) ]
                    results_marginal = np.array(results_marginal)
                    
                    if np.any(results_marginal==1):
                        indexes_to_be_removed.append(ind_small)
            
                results = np.array(results)
                
                if np.any(results==1):
                    indexes_to_be_removed.append(ind_small)
                
            
            if len(indexes_to_be_removed)>0:
                indexes_to_be_removed = np.unique(indexes_to_be_removed)
                indexes_to_be_removed = np.sort(indexes_to_be_removed)[::-1]
                for ind in indexes_to_be_removed:
                    contours.pop(ind)

            return contours
                    
                
        else:
            contours_txtline_of_all_textregions = []
            indexes_of_textline_tot = []
            index_textline_inside_textregion = []
            
            for jj in range(len(contours)):
                contours_txtline_of_all_textregions = contours_txtline_of_all_textregions + contours[jj]
                
                ind_ins = np.zeros( len(contours[jj]) ) + jj
                list_ind_ins = list(ind_ins)
                
                ind_textline_inside_tr = np.array (range(len(contours[jj])) )
                
                list_ind_textline_inside_tr = list(ind_textline_inside_tr)
                                                  
                index_textline_inside_textregion = index_textline_inside_textregion + list_ind_textline_inside_tr
                
                indexes_of_textline_tot = indexes_of_textline_tot + list_ind_ins
                
                
            M_main_tot = [cv2.moments(contours_txtline_of_all_textregions[j]) for j in range(len(contours_txtline_of_all_textregions))]
            cx_main_tot = [(M_main_tot[j]["m10"] / (M_main_tot[j]["m00"] + 1e-32)) for j in range(len(M_main_tot))]
            cy_main_tot = [(M_main_tot[j]["m01"] / (M_main_tot[j]["m00"] + 1e-32)) for j in range(len(M_main_tot))]
            
            
            areas_tot = [cv2.contourArea(con_ind) for con_ind in contours_txtline_of_all_textregions]
            area_tot_tot = image.shape[0]*image.shape[1]
            
            textregion_index_to_del = []
            textline_in_textregion_index_to_del = []
            for ij in range(len(contours_txtline_of_all_textregions)):
                
                args_all = list(np.array(range(len(contours_txtline_of_all_textregions))))
                
                args_all.pop(ij)
                
                areas_without = np.array(areas_tot)[args_all]
                area_of_con_interest = areas_tot[ij]
                
                args_with_bigger_area = np.array(args_all)[areas_without > 1.5*area_of_con_interest]
                
                if len(args_with_bigger_area)>0:
                    results = [cv2.pointPolygonTest(contours_txtline_of_all_textregions[ind], (cx_main_tot[ij], cy_main_tot[ij]), False) for ind in args_with_bigger_area ]
                    results = np.array(results)
                    if np.any(results==1):
                        #print(indexes_of_textline_tot[ij], index_textline_inside_textregion[ij])
                        textregion_index_to_del.append(int(indexes_of_textline_tot[ij]))
                        textline_in_textregion_index_to_del.append(int(index_textline_inside_textregion[ij]))
                        #contours[int(indexes_of_textline_tot[ij])].pop(int(index_textline_inside_textregion[ij]))
                        
            uniqe_args_trs = np.unique(textregion_index_to_del)
            
            for ind_u_a_trs in uniqe_args_trs:
                textline_in_textregion_index_to_del_ind = np.array(textline_in_textregion_index_to_del)[np.array(textregion_index_to_del)==ind_u_a_trs]
                textline_in_textregion_index_to_del_ind = np.sort(textline_in_textregion_index_to_del_ind)[::-1]
                
                for ittrd in textline_in_textregion_index_to_del_ind:
                    contours[ind_u_a_trs].pop(ittrd)
                        
            return contours
        
            
                    
                    
        
    
    def dilate_textlines(self,all_found_textline_polygons):
        for j in range(len(all_found_textline_polygons)):
            for i in range(len(all_found_textline_polygons[j])):
                con_ind = all_found_textline_polygons[j][i]
                
                con_ind = con_ind.astype(np.float)
                
                x_differential = np.diff( con_ind[:,0,0])
                y_differential = np.diff( con_ind[:,0,1])
                
                x_min = float(np.min( con_ind[:,0,0] ))
                y_min = float(np.min( con_ind[:,0,1] ))
                
                x_max = float(np.max( con_ind[:,0,0] ))
                y_max = float(np.max( con_ind[:,0,1] ))

                
                if (y_max - y_min) > (x_max - x_min) and (x_max - x_min)<70:
                    
                    x_biger_than_x = np.abs(x_differential) > np.abs(y_differential)
                    
                    mult = x_biger_than_x*x_differential
                    
                    arg_min_mult = np.argmin(mult)
                    arg_max_mult = np.argmax(mult)
                    
                    if y_differential[0]==0:
                        y_differential[0] = 0.1
                    
                    if y_differential[-1]==0:
                        y_differential[-1]= 0.1
                        
                        
                        
                    y_differential = [y_differential[ind] if y_differential[ind]!=0 else (y_differential[ind-1] + y_differential[ind+1])/2. for ind in range(len(y_differential)) ]
                    
                    
                    if y_differential[0]==0.1:
                        y_differential[0] = y_differential[1]
                    if y_differential[-1]==0.1:
                        y_differential[-1] = y_differential[-2]
                        
                    y_differential.append(y_differential[0])
                    
                    y_differential = [-1 if y_differential[ind]<0 else 1 for ind in range(len(y_differential))]
                    
                    y_differential = self.return_it_in_two_groups(y_differential)
                    
                    y_differential = np.array(y_differential)
                    
                    
                    con_scaled = con_ind*1
                    
                    con_scaled[:,0, 0] = con_ind[:,0,0] - 8*y_differential
                    
                    con_scaled[arg_min_mult,0, 1] = con_ind[arg_min_mult,0,1] + 8
                    con_scaled[arg_min_mult+1,0, 1] = con_ind[arg_min_mult+1,0,1] + 8
                    
                    try:
                        con_scaled[arg_min_mult-1,0, 1] = con_ind[arg_min_mult-1,0,1] + 5
                        con_scaled[arg_min_mult+2,0, 1] = con_ind[arg_min_mult+2,0,1] + 5
                    except:
                        pass
                    
                    con_scaled[arg_max_mult,0, 1] = con_ind[arg_max_mult,0,1] - 8
                    con_scaled[arg_max_mult+1,0, 1] = con_ind[arg_max_mult+1,0,1] - 8
                    
                    try:
                        con_scaled[arg_max_mult-1,0, 1] = con_ind[arg_max_mult-1,0,1] - 5
                        con_scaled[arg_max_mult+2,0, 1] = con_ind[arg_max_mult+2,0,1] - 5
                    except:
                        pass
                
                
                else:
                    y_biger_than_x = np.abs(y_differential) > np.abs(x_differential)
                    
                    mult = y_biger_than_x*y_differential
                    
                    arg_min_mult = np.argmin(mult)
                    arg_max_mult = np.argmax(mult)
                    
                    if x_differential[0]==0:
                        x_differential[0] = 0.1
                    
                    if x_differential[-1]==0:
                        x_differential[-1]= 0.1
                        
                        
                        
                    x_differential = [x_differential[ind] if x_differential[ind]!=0 else (x_differential[ind-1] + x_differential[ind+1])/2. for ind in range(len(x_differential)) ]
                    
                    
                    if x_differential[0]==0.1:
                        x_differential[0] = x_differential[1]
                    if x_differential[-1]==0.1:
                        x_differential[-1] = x_differential[-2]
                        
                    x_differential.append(x_differential[0])
                    
                    x_differential = [-1 if x_differential[ind]<0 else 1 for ind in range(len(x_differential))]
                    
                    x_differential = self.return_it_in_two_groups(x_differential)
                    x_differential = np.array(x_differential)
                    
                    
                    con_scaled = con_ind*1
                    
                    con_scaled[:,0, 1] = con_ind[:,0,1] + 8*x_differential
                    
                    con_scaled[arg_min_mult,0, 0] = con_ind[arg_min_mult,0,0] + 8
                    con_scaled[arg_min_mult+1,0, 0] = con_ind[arg_min_mult+1,0,0] + 8
                    
                    try:
                        con_scaled[arg_min_mult-1,0, 0] = con_ind[arg_min_mult-1,0,0] + 5
                        con_scaled[arg_min_mult+2,0, 0] = con_ind[arg_min_mult+2,0,0] + 5
                    except:
                        pass
                    
                    con_scaled[arg_max_mult,0, 0] = con_ind[arg_max_mult,0,0] - 8
                    con_scaled[arg_max_mult+1,0, 0] = con_ind[arg_max_mult+1,0,0] - 8
                    
                    try:
                        con_scaled[arg_max_mult-1,0, 0] = con_ind[arg_max_mult-1,0,0] - 5
                        con_scaled[arg_max_mult+2,0, 0] = con_ind[arg_max_mult+2,0,0] - 5
                    except:
                        pass
                    
                
                con_scaled[:,0, 1][con_scaled[:,0, 1]<0] = 0
                con_scaled[:,0, 0][con_scaled[:,0, 0]<0] = 0
                
                all_found_textline_polygons[j][i][:,0,1] = con_scaled[:,0, 1]
                all_found_textline_polygons[j][i][:,0,0] = con_scaled[:,0, 0]
            
        return all_found_textline_polygons
    
    def delete_regions_without_textlines(self,slopes, all_found_textline_polygons, boxes_text, txt_con_org, contours_only_text_parent, index_by_text_par_con):
        slopes_rem = []
        all_found_textline_polygons_rem = []
        boxes_text_rem = []
        txt_con_org_rem = []
        contours_only_text_parent_rem = []
        index_by_text_par_con_rem = []
        
        for i, ind_con in enumerate(all_found_textline_polygons):
            if len(ind_con):
                all_found_textline_polygons_rem.append(ind_con)
                slopes_rem.append(slopes[i])
                boxes_text_rem.append(boxes_text[i])
                txt_con_org_rem.append(txt_con_org[i])
                contours_only_text_parent_rem.append(contours_only_text_parent[i])
                index_by_text_par_con_rem.append(index_by_text_par_con[i])
                
        index_sort = np.argsort(index_by_text_par_con_rem)
        indexes_new = np.array(range(len(index_by_text_par_con_rem)))
        
        index_by_text_par_con_rem_sort = [indexes_new[index_sort==j][0] for j in range(len(index_by_text_par_con_rem))]
                
        return slopes_rem, all_found_textline_polygons_rem, boxes_text_rem, txt_con_org_rem, contours_only_text_parent_rem, index_by_text_par_con_rem_sort

    def run(self):
        """
        Get image and scales, then extract the page of scanned image
        """
        self.logger.debug("enter run")

        t0_tot = time.time()

        if not self.dir_in:
            self.ls_imgs = [1]
        
        for img_name in self.ls_imgs:
            print(img_name)
            t0 = time.time()
            if self.dir_in:
                self.reset_file_name_dir(os.path.join(self.dir_in,img_name))
                #print("text region early -11 in %.1fs", time.time() - t0)
                
                
            if self.extract_only_images:
                img_res, is_image_enhanced, num_col_classifier, num_column_is_classified = self.run_enhancement(self.light_version)
                self.logger.info("Enhancing took %.1fs ", time.time() - t0)

                text_regions_p_1 ,erosion_hurts, polygons_lines_xml,polygons_of_images,image_page, page_coord, cont_page = self.get_regions_light_v_extract_only_images(img_res, is_image_enhanced, num_col_classifier)
                ocr_all_textlines = None
                pcgts = self.writer.build_pagexml_no_full_layout([], page_coord, [], [], [], [], polygons_of_images, [], [], [], [], [], cont_page, [], [], ocr_all_textlines)

                if self.plotter:
                    self.plotter.write_images_into_directory(polygons_of_images, image_page)

                if self.dir_in:
                    self.writer.write_pagexml(pcgts)
                else:
                    return pcgts
            else:
                img_res, is_image_enhanced, num_col_classifier, num_column_is_classified = self.run_enhancement(self.light_version)
                self.logger.info("Enhancing took %.1fs ", time.time() - t0)
                #print("text region early -1 in %.1fs", time.time() - t0)
                t1 = time.time()
                if not self.skip_layout_and_reading_order:
                    if self.light_version:
                        text_regions_p_1 ,erosion_hurts, polygons_lines_xml, textline_mask_tot_ea, img_bin_light = self.get_regions_light_v(img_res, is_image_enhanced, num_col_classifier)
                        #print("text region early -2 in %.1fs", time.time() - t0)
                        
                        if num_col_classifier == 1 or num_col_classifier ==2:
                            if num_col_classifier == 1:
                                img_w_new = 1000
                                img_h_new = int(textline_mask_tot_ea.shape[0] / float(textline_mask_tot_ea.shape[1]) * img_w_new)
                                
                            elif num_col_classifier == 2:
                                img_w_new = 1300
                                img_h_new = int(textline_mask_tot_ea.shape[0] / float(textline_mask_tot_ea.shape[1]) * img_w_new)
                                
                            textline_mask_tot_ea_deskew = resize_image(textline_mask_tot_ea,img_h_new, img_w_new )
                            
                            slope_deskew, slope_first = self.run_deskew(textline_mask_tot_ea_deskew)
                        else:
                            slope_deskew, slope_first = self.run_deskew(textline_mask_tot_ea)
                        #print("text region early -2,5 in %.1fs", time.time() - t0)
                        #self.logger.info("Textregion detection took %.1fs ", time.time() - t1t)
                        num_col, num_col_classifier, img_only_regions, page_coord, image_page, mask_images, mask_lines, text_regions_p_1, cont_page, table_prediction, textline_mask_tot_ea, img_bin_light = \
                                self.run_graphics_and_columns_light(text_regions_p_1, textline_mask_tot_ea, num_col_classifier, num_column_is_classified, erosion_hurts, img_bin_light)
                        #self.logger.info("run graphics %.1fs ", time.time() - t1t)
                        #print("text region early -3 in %.1fs", time.time() - t0)
                        textline_mask_tot_ea_org = np.copy(textline_mask_tot_ea)
                        #print("text region early -4 in %.1fs", time.time() - t0)
                    else:
                        text_regions_p_1 ,erosion_hurts, polygons_lines_xml = self.get_regions_from_xy_2models(img_res, is_image_enhanced, num_col_classifier)
                        self.logger.info("Textregion detection took %.1fs ", time.time() - t1)

                        t1 = time.time()
                        num_col, num_col_classifier, img_only_regions, page_coord, image_page, mask_images, mask_lines, text_regions_p_1, cont_page, table_prediction = \
                                self.run_graphics_and_columns(text_regions_p_1, num_col_classifier, num_column_is_classified, erosion_hurts)
                        self.logger.info("Graphics detection took %.1fs ", time.time() - t1)
                        #self.logger.info('cont_page %s', cont_page)
                    
                    if not num_col:
                        self.logger.info("No columns detected, outputting an empty PAGE-XML")
                        ocr_all_textlines = None
                        pcgts = self.writer.build_pagexml_no_full_layout([], page_coord, [], [], [], [], [], [], [], [], [], [], cont_page, [], [], ocr_all_textlines)
                        self.logger.info("Job done in %.1fs", time.time() - t1)
                        if self.dir_in:
                            self.writer.write_pagexml(pcgts)
                            continue
                        else:
                            return pcgts
                    #print("text region early in %.1fs", time.time() - t0)
                    t1 = time.time()
                    if not self.light_version:
                        textline_mask_tot_ea = self.run_textline(image_page)
                        self.logger.info("textline detection took %.1fs", time.time() - t1)

                        t1 = time.time()
                        slope_deskew, slope_first = self.run_deskew(textline_mask_tot_ea)
                        self.logger.info("deskewing took %.1fs", time.time() - t1)
                    t1 = time.time()
                    #plt.imshow(table_prediction)
                    #plt.show()
                    if self.light_version and num_col_classifier in (1,2):
                        org_h_l_m = textline_mask_tot_ea.shape[0]
                        org_w_l_m = textline_mask_tot_ea.shape[1]
                        if num_col_classifier == 1:
                            img_w_new = 2000
                            img_h_new = int(textline_mask_tot_ea.shape[0] / float(textline_mask_tot_ea.shape[1]) * img_w_new)
                            
                        elif num_col_classifier == 2:
                            img_w_new = 2400
                            img_h_new = int(textline_mask_tot_ea.shape[0] / float(textline_mask_tot_ea.shape[1]) * img_w_new)
                            
                        image_page = resize_image(image_page,img_h_new, img_w_new )
                        textline_mask_tot_ea = resize_image(textline_mask_tot_ea,img_h_new, img_w_new )
                        mask_images = resize_image(mask_images,img_h_new, img_w_new )
                        mask_lines = resize_image(mask_lines,img_h_new, img_w_new )
                        text_regions_p_1 = resize_image(text_regions_p_1,img_h_new, img_w_new )
                        table_prediction = resize_image(table_prediction,img_h_new, img_w_new )
                        
                    textline_mask_tot, text_regions_p, image_page_rotated = self.run_marginals(image_page, textline_mask_tot_ea, mask_images, mask_lines, num_col_classifier, slope_deskew, text_regions_p_1, table_prediction)
                    
                    if self.light_version and num_col_classifier in (1,2):
                        image_page = resize_image(image_page,org_h_l_m, org_w_l_m )
                        textline_mask_tot_ea = resize_image(textline_mask_tot_ea,org_h_l_m, org_w_l_m )
                        text_regions_p = resize_image(text_regions_p,org_h_l_m, org_w_l_m )
                        textline_mask_tot = resize_image(textline_mask_tot,org_h_l_m, org_w_l_m )
                        text_regions_p_1 = resize_image(text_regions_p_1,org_h_l_m, org_w_l_m )
                        table_prediction = resize_image(table_prediction,org_h_l_m, org_w_l_m )
                        image_page_rotated = resize_image(image_page_rotated,org_h_l_m, org_w_l_m )
                        
                    self.logger.info("detection of marginals took %.1fs", time.time() - t1)
                    #print("text region early 2 marginal in %.1fs", time.time() - t0)
                    ## birdan sora chock chakir
                    t1 = time.time()
                    if not self.full_layout:
                        polygons_of_images, img_revised_tab, text_regions_p_1_n, textline_mask_tot_d, regions_without_separators_d, boxes, boxes_d, polygons_of_marginals, contours_tables = self.run_boxes_no_full_layout(image_page, textline_mask_tot, text_regions_p, slope_deskew, num_col_classifier, table_prediction, erosion_hurts)
                        ###polygons_of_marginals = self.dilate_textregions_contours(polygons_of_marginals)
                    if self.full_layout:
                        if not self.light_version:
                            img_bin_light = None
                        polygons_of_images, img_revised_tab, text_regions_p_1_n, textline_mask_tot_d, regions_without_separators_d, regions_fully, regions_without_separators, polygons_of_marginals, contours_tables = self.run_boxes_full_layout(image_page, textline_mask_tot, text_regions_p, slope_deskew, num_col_classifier, img_only_regions, table_prediction, erosion_hurts, img_bin_light)
                        ###polygons_of_marginals = self.dilate_textregions_contours(polygons_of_marginals)
                        
                        if self.light_version:
                            drop_label_in_full_layout = 4
                            textline_mask_tot_ea_org[img_revised_tab==drop_label_in_full_layout] = 0
                            
                        
                    text_only = ((img_revised_tab[:, :] == 1)) * 1
                    if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                        text_only_d = ((text_regions_p_1_n[:, :] == 1)) * 1
                    
                    #print("text region early 2 in %.1fs", time.time() - t0)
                    ###min_con_area = 0.000005
                    if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                        contours_only_text, hir_on_text = return_contours_of_image(text_only)
                        contours_only_text_parent = return_parent_contours(contours_only_text, hir_on_text)
                                    
                        if len(contours_only_text_parent) > 0:
                            areas_cnt_text = np.array([cv2.contourArea(c) for c in contours_only_text_parent])
                            areas_cnt_text = areas_cnt_text / float(text_only.shape[0] * text_only.shape[1])
                            #self.logger.info('areas_cnt_text %s', areas_cnt_text)
                            contours_biggest = contours_only_text_parent[np.argmax(areas_cnt_text)]
                            contours_only_text_parent = [c for jz, c in enumerate(contours_only_text_parent) if areas_cnt_text[jz] > MIN_AREA_REGION]
                            areas_cnt_text_parent = [area for area in areas_cnt_text if area > MIN_AREA_REGION]
                            index_con_parents = np.argsort(areas_cnt_text_parent)
                            
                            contours_only_text_parent = self.return_list_of_contours_with_desired_order(contours_only_text_parent, index_con_parents)

                            ##try:
                                ##contours_only_text_parent = list(np.array(contours_only_text_parent,dtype=object)[index_con_parents])
                            ##except:
                                ##contours_only_text_parent = list(np.array(contours_only_text_parent,dtype=np.int32)[index_con_parents])
                            ##areas_cnt_text_parent = list(np.array(areas_cnt_text_parent)[index_con_parents])
                            areas_cnt_text_parent = self.return_list_of_contours_with_desired_order(areas_cnt_text_parent, index_con_parents)

                            cx_bigest_big, cy_biggest_big, _, _, _, _, _ = find_new_features_of_contours([contours_biggest])
                            cx_bigest, cy_biggest, _, _, _, _, _ = find_new_features_of_contours(contours_only_text_parent)

                            contours_only_text_d, hir_on_text_d = return_contours_of_image(text_only_d)
                            contours_only_text_parent_d = return_parent_contours(contours_only_text_d, hir_on_text_d)

                            areas_cnt_text_d = np.array([cv2.contourArea(c) for c in contours_only_text_parent_d])
                            areas_cnt_text_d = areas_cnt_text_d / float(text_only_d.shape[0] * text_only_d.shape[1])
                            
                            if len(areas_cnt_text_d)>0:
                                contours_biggest_d = contours_only_text_parent_d[np.argmax(areas_cnt_text_d)]
                                index_con_parents_d = np.argsort(areas_cnt_text_d)
                                contours_only_text_parent_d = self.return_list_of_contours_with_desired_order(contours_only_text_parent_d, index_con_parents_d)
                                #try:
                                    #contours_only_text_parent_d = list(np.array(contours_only_text_parent_d,dtype=object)[index_con_parents_d])
                                #except:
                                    #contours_only_text_parent_d = list(np.array(contours_only_text_parent_d,dtype=np.int32)[index_con_parents_d])
                                    
                                #areas_cnt_text_d = list(np.array(areas_cnt_text_d)[index_con_parents_d])
                                areas_cnt_text_d = self.return_list_of_contours_with_desired_order(areas_cnt_text_d, index_con_parents_d)

                                cx_bigest_d_big, cy_biggest_d_big, _, _, _, _, _ = find_new_features_of_contours([contours_biggest_d])
                                cx_bigest_d, cy_biggest_d, _, _, _, _, _ = find_new_features_of_contours(contours_only_text_parent_d)
                                try:
                                    if len(cx_bigest_d) >= 5:
                                        cx_bigest_d_last5 = cx_bigest_d[-5:]
                                        cy_biggest_d_last5 = cy_biggest_d[-5:]
                                        dists_d = [math.sqrt((cx_bigest_big[0] - cx_bigest_d_last5[j]) ** 2 + (cy_biggest_big[0] - cy_biggest_d_last5[j]) ** 2) for j in range(len(cy_biggest_d_last5))]
                                        ind_largest = len(cx_bigest_d) -5 + np.argmin(dists_d)
                                    else:
                                        cx_bigest_d_last5 = cx_bigest_d[-len(cx_bigest_d):]
                                        cy_biggest_d_last5 = cy_biggest_d[-len(cx_bigest_d):]
                                        dists_d = [math.sqrt((cx_bigest_big[0]-cx_bigest_d_last5[j])**2 + (cy_biggest_big[0]-cy_biggest_d_last5[j])**2) for j in range(len(cy_biggest_d_last5))]
                                        ind_largest = len(cx_bigest_d) - len(cx_bigest_d) + np.argmin(dists_d)
                                        
                                    cx_bigest_d_big[0] = cx_bigest_d[ind_largest]
                                    cy_biggest_d_big[0] = cy_biggest_d[ind_largest]
                                except Exception as why:
                                    self.logger.error(why)

                                (h, w) = text_only.shape[:2]
                                center = (w // 2.0, h // 2.0)
                                M = cv2.getRotationMatrix2D(center, slope_deskew, 1.0)
                                M_22 = np.array(M)[:2, :2]
                                p_big = np.dot(M_22, [cx_bigest_big, cy_biggest_big])
                                x_diff = p_big[0] - cx_bigest_d_big
                                y_diff = p_big[1] - cy_biggest_d_big

                                contours_only_text_parent_d_ordered = []
                                for i in range(len(contours_only_text_parent)):
                                    p = np.dot(M_22, [cx_bigest[i], cy_biggest[i]])
                                    p[0] = p[0] - x_diff[0]
                                    p[1] = p[1] - y_diff[0]
                                    dists = [math.sqrt((p[0] - cx_bigest_d[j]) ** 2 + (p[1] - cy_biggest_d[j]) ** 2) for j in range(len(cx_bigest_d))]
                                    contours_only_text_parent_d_ordered.append(contours_only_text_parent_d[np.argmin(dists)])
                                    # img2=np.zeros((text_only.shape[0],text_only.shape[1],3))
                                    # img2=cv2.fillPoly(img2,pts=[contours_only_text_parent_d[np.argmin(dists)]] ,color=(1,1,1))
                                    # plt.imshow(img2[:,:,0])
                                    # plt.show()
                            else:
                                contours_only_text_parent_d_ordered = []
                                contours_only_text_parent_d = []
                                contours_only_text_parent = []
                                
                        else:
                            contours_only_text_parent_d_ordered = []
                            contours_only_text_parent_d = []
                            contours_only_text_parent = []
                    else:
                        contours_only_text, hir_on_text = return_contours_of_image(text_only)
                        contours_only_text_parent = return_parent_contours(contours_only_text, hir_on_text)
                        
                        if len(contours_only_text_parent) > 0:
                            areas_cnt_text = np.array([cv2.contourArea(c) for c in contours_only_text_parent])
                            areas_cnt_text = areas_cnt_text / float(text_only.shape[0] * text_only.shape[1])

                            contours_biggest = contours_only_text_parent[np.argmax(areas_cnt_text)]
                            contours_only_text_parent = [c for jz, c in enumerate(contours_only_text_parent) if areas_cnt_text[jz] > MIN_AREA_REGION]
                            areas_cnt_text_parent = [area for area in areas_cnt_text if area > MIN_AREA_REGION]

                            index_con_parents = np.argsort(areas_cnt_text_parent)
                            
                            contours_only_text_parent = self.return_list_of_contours_with_desired_order(contours_only_text_parent, index_con_parents)
                            #try:
                                #contours_only_text_parent = list(np.array(contours_only_text_parent,dtype=object)[index_con_parents])
                            #except:
                                #contours_only_text_parent = list(np.array(contours_only_text_parent,dtype=np.int32)[index_con_parents])
                            #areas_cnt_text_parent = list(np.array(areas_cnt_text_parent)[index_con_parents])
                            areas_cnt_text_parent = self.return_list_of_contours_with_desired_order(areas_cnt_text_parent, index_con_parents)

                            cx_bigest_big, cy_biggest_big, _, _, _, _, _ = find_new_features_of_contours([contours_biggest])
                            cx_bigest, cy_biggest, _, _, _, _, _ = find_new_features_of_contours(contours_only_text_parent)
                            #self.logger.debug('areas_cnt_text_parent %s', areas_cnt_text_parent)
                            # self.logger.debug('areas_cnt_text_parent_d %s', areas_cnt_text_parent_d)
                            # self.logger.debug('len(contours_only_text_parent) %s', len(contours_only_text_parent_d))
                        else:
                            pass
                        
                    #print("text region early 3 in %.1fs", time.time() - t0)
                    if self.light_version:
                        contours_only_text_parent = self.dilate_textregions_contours(contours_only_text_parent)
                        contours_only_text_parent = self.filter_contours_inside_a_bigger_one(contours_only_text_parent, text_only, marginal_cnts=polygons_of_marginals)
                        #print("text region early 3.5 in %.1fs", time.time() - t0)
                        txt_con_org = get_textregion_contours_in_org_image_light(contours_only_text_parent, self.image, slope_first)
                        #txt_con_org = self.dilate_textregions_contours(txt_con_org)
                        #contours_only_text_parent = self.dilate_textregions_contours(contours_only_text_parent)
                    else:
                        txt_con_org = get_textregion_contours_in_org_image(contours_only_text_parent, self.image, slope_first)
                    #print("text region early 4 in %.1fs", time.time() - t0)
                    boxes_text, _ = get_text_region_boxes_by_given_contours(contours_only_text_parent)
                    boxes_marginals, _ = get_text_region_boxes_by_given_contours(polygons_of_marginals)
                    #print("text region early 5 in %.1fs", time.time() - t0)
                    ## birdan sora chock chakir
                    if not self.curved_line:
                        if self.light_version:
                            if self.textline_light:
                                #slopes, all_found_textline_polygons, boxes_text, txt_con_org, contours_only_text_parent, all_box_coord, index_by_text_par_con = self.get_slopes_and_deskew_new_light(txt_con_org, contours_only_text_parent, textline_mask_tot_ea_org, image_page_rotated, boxes_text, slope_deskew)
                                
                                slopes, all_found_textline_polygons, boxes_text, txt_con_org, contours_only_text_parent, all_box_coord, index_by_text_par_con = self.get_slopes_and_deskew_new_light2(txt_con_org, contours_only_text_parent, textline_mask_tot_ea_org, image_page_rotated, boxes_text, slope_deskew)
                                slopes_marginals, all_found_textline_polygons_marginals, boxes_marginals, _, polygons_of_marginals, all_box_coord_marginals, _ = self.get_slopes_and_deskew_new_light(polygons_of_marginals, polygons_of_marginals, textline_mask_tot_ea_org, image_page_rotated, boxes_marginals, slope_deskew)
                                
                                #slopes, all_found_textline_polygons, boxes_text, txt_con_org, contours_only_text_parent, index_by_text_par_con = self.delete_regions_without_textlines(slopes, all_found_textline_polygons, boxes_text, txt_con_org, contours_only_text_parent, index_by_text_par_con)
                                
                                #slopes_marginals, all_found_textline_polygons_marginals, boxes_marginals, polygons_of_marginals, polygons_of_marginals, _ = self.delete_regions_without_textlines(slopes_marginals, all_found_textline_polygons_marginals, boxes_marginals, polygons_of_marginals, polygons_of_marginals, np.array(range(len(polygons_of_marginals))))
                                #all_found_textline_polygons = self.dilate_textlines(all_found_textline_polygons)
                                #####all_found_textline_polygons = self.dilate_textline_contours(all_found_textline_polygons)
                                all_found_textline_polygons = self.dilate_textregions_contours_textline_version(all_found_textline_polygons)
                                all_found_textline_polygons = self.filter_contours_inside_a_bigger_one(all_found_textline_polygons, textline_mask_tot_ea_org, type_contour="textline")
                                all_found_textline_polygons_marginals = self.dilate_textregions_contours_textline_version(all_found_textline_polygons_marginals)
                                
                            else:
                                textline_mask_tot_ea = cv2.erode(textline_mask_tot_ea, kernel=KERNEL, iterations=1)
                                slopes, all_found_textline_polygons, boxes_text, txt_con_org, contours_only_text_parent, all_box_coord, index_by_text_par_con = self.get_slopes_and_deskew_new_light(txt_con_org, contours_only_text_parent, textline_mask_tot_ea, image_page_rotated, boxes_text, slope_deskew)
                                slopes_marginals, all_found_textline_polygons_marginals, boxes_marginals, _, polygons_of_marginals, all_box_coord_marginals, _ = self.get_slopes_and_deskew_new_light(polygons_of_marginals, polygons_of_marginals, textline_mask_tot_ea, image_page_rotated, boxes_marginals, slope_deskew)
                                
                                #all_found_textline_polygons = self.filter_contours_inside_a_bigger_one(all_found_textline_polygons, textline_mask_tot_ea_org, type_contour="textline")
                        else:
                            textline_mask_tot_ea = cv2.erode(textline_mask_tot_ea, kernel=KERNEL, iterations=1)
                            slopes, all_found_textline_polygons, boxes_text, txt_con_org, contours_only_text_parent, all_box_coord, index_by_text_par_con = self.get_slopes_and_deskew_new(txt_con_org, contours_only_text_parent, textline_mask_tot_ea, image_page_rotated, boxes_text, slope_deskew)
                            slopes_marginals, all_found_textline_polygons_marginals, boxes_marginals, _, polygons_of_marginals, all_box_coord_marginals, _ = self.get_slopes_and_deskew_new(polygons_of_marginals, polygons_of_marginals, textline_mask_tot_ea, image_page_rotated, boxes_marginals, slope_deskew)

                    else:
                        
                        scale_param = 1
                        all_found_textline_polygons, boxes_text, txt_con_org, contours_only_text_parent, all_box_coord, index_by_text_par_con, slopes = self.get_slopes_and_deskew_new_curved(txt_con_org, contours_only_text_parent, cv2.erode(textline_mask_tot_ea, kernel=KERNEL, iterations=2), image_page_rotated, boxes_text, text_only, num_col_classifier, scale_param, slope_deskew)
                        all_found_textline_polygons = small_textlines_to_parent_adherence2(all_found_textline_polygons, textline_mask_tot_ea, num_col_classifier)
                        all_found_textline_polygons_marginals, boxes_marginals, _, polygons_of_marginals, all_box_coord_marginals, _, slopes_marginals = self.get_slopes_and_deskew_new_curved(polygons_of_marginals, polygons_of_marginals, cv2.erode(textline_mask_tot_ea, kernel=KERNEL, iterations=2), image_page_rotated, boxes_marginals, text_only, num_col_classifier, scale_param, slope_deskew)
                        all_found_textline_polygons_marginals = small_textlines_to_parent_adherence2(all_found_textline_polygons_marginals, textline_mask_tot_ea, num_col_classifier)
                    #print("text region early 6 in %.1fs", time.time() - t0)
                    if self.full_layout:
                        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                            contours_only_text_parent_d_ordered = self.return_list_of_contours_with_desired_order(contours_only_text_parent_d_ordered, index_by_text_par_con)
                            #try:
                                #contours_only_text_parent_d_ordered = list(np.array(contours_only_text_parent_d_ordered, dtype=np.int32)[index_by_text_par_con])
                            #except:
                                #contours_only_text_parent_d_ordered = list(np.array(contours_only_text_parent_d_ordered, dtype=object)[index_by_text_par_con])
                            if self.light_version:
                                text_regions_p, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, all_found_textline_polygons, all_found_textline_polygons_h, slopes, slopes_h, contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered = check_any_text_region_in_model_one_is_main_or_header_light(text_regions_p, regions_fully, contours_only_text_parent, all_box_coord, all_found_textline_polygons, slopes, contours_only_text_parent_d_ordered)
                            else:
                                text_regions_p, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, all_found_textline_polygons, all_found_textline_polygons_h, slopes, slopes_h, contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered = check_any_text_region_in_model_one_is_main_or_header(text_regions_p, regions_fully, contours_only_text_parent, all_box_coord, all_found_textline_polygons, slopes, contours_only_text_parent_d_ordered)
                        else:
                            #takes long timee
                            contours_only_text_parent_d_ordered = None
                            if self.light_version:
                                text_regions_p, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, all_found_textline_polygons, all_found_textline_polygons_h, slopes, slopes_h, contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered = check_any_text_region_in_model_one_is_main_or_header_light(text_regions_p, regions_fully, contours_only_text_parent, all_box_coord, all_found_textline_polygons, slopes, contours_only_text_parent_d_ordered)
                            else:
                                text_regions_p, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, all_found_textline_polygons, all_found_textline_polygons_h, slopes, slopes_h, contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered = check_any_text_region_in_model_one_is_main_or_header(text_regions_p, regions_fully, contours_only_text_parent, all_box_coord, all_found_textline_polygons, slopes, contours_only_text_parent_d_ordered)

                        if self.plotter:
                            self.plotter.save_plot_of_layout(text_regions_p, image_page)
                            self.plotter.save_plot_of_layout_all(text_regions_p, image_page)
                
                        pixel_img = 4
                        polygons_of_drop_capitals = return_contours_of_interested_region_by_min_size(text_regions_p, pixel_img)
                        all_found_textline_polygons = adhere_drop_capital_region_into_corresponding_textline(text_regions_p, polygons_of_drop_capitals, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, all_found_textline_polygons, all_found_textline_polygons_h, kernel=KERNEL, curved_line=self.curved_line, textline_light=self.textline_light)
                        pixel_lines = 6
                        
                        if not self.reading_order_machine_based:
                            if not self.headers_off:
                                if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                                    num_col, _, matrix_of_lines_ch, splitter_y_new, _ = find_number_of_columns_in_document(np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2), num_col_classifier, self.tables,  pixel_lines, contours_only_text_parent_h)
                                else:
                                    _, _, matrix_of_lines_ch_d, splitter_y_new_d, _ = find_number_of_columns_in_document(np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2), num_col_classifier, self.tables, pixel_lines, contours_only_text_parent_h_d_ordered)
                            elif self.headers_off:
                                if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                                    num_col, _, matrix_of_lines_ch, splitter_y_new, _ = find_number_of_columns_in_document(np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2), num_col_classifier, self.tables,  pixel_lines)
                                else:
                                    _, _, matrix_of_lines_ch_d, splitter_y_new_d, _ = find_number_of_columns_in_document(np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2), num_col_classifier, self.tables, pixel_lines)

                            if num_col_classifier >= 3:
                                if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                                    regions_without_separators = regions_without_separators.astype(np.uint8)
                                    regions_without_separators = cv2.erode(regions_without_separators[:, :], KERNEL, iterations=6)

                                else:
                                    regions_without_separators_d = regions_without_separators_d.astype(np.uint8)
                                    regions_without_separators_d = cv2.erode(regions_without_separators_d[:, :], KERNEL, iterations=6)
                                
                        if not self.reading_order_machine_based:
                            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                                boxes, peaks_neg_tot_tables = return_boxes_of_images_by_order_of_reading_new(splitter_y_new, regions_without_separators, matrix_of_lines_ch, num_col_classifier, erosion_hurts, self.tables, self.right2left)
                            else:
                                boxes_d, peaks_neg_tot_tables_d = return_boxes_of_images_by_order_of_reading_new(splitter_y_new_d, regions_without_separators_d, matrix_of_lines_ch_d, num_col_classifier, erosion_hurts, self.tables, self.right2left)     

                    if self.plotter:
                        self.plotter.write_images_into_directory(polygons_of_images, image_page)
                    t_order = time.time()
                            
                    if self.full_layout:
                        
                        if self.reading_order_machine_based:
                            order_text_new, id_of_texts_tot = self.do_order_of_regions_with_machine_optimized_algorithm(contours_only_text_parent, contours_only_text_parent_h, text_regions_p)
                        else:
                            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                                order_text_new, id_of_texts_tot = self.do_order_of_regions(contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot)
                            else:
                                order_text_new, id_of_texts_tot = self.do_order_of_regions(contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered, boxes_d, textline_mask_tot_d)
                        self.logger.info("detection of reading order took %.1fs", time.time() - t_order)
                        
                        if self.ocr:
                            ocr_all_textlines = []
                        else:
                            ocr_all_textlines = None
                            
                        pcgts = self.writer.build_pagexml_full_layout(contours_only_text_parent, contours_only_text_parent_h, page_coord, order_text_new, id_of_texts_tot, all_found_textline_polygons, all_found_textline_polygons_h, all_box_coord, all_box_coord_h, polygons_of_images, contours_tables, polygons_of_drop_capitals, polygons_of_marginals, all_found_textline_polygons_marginals, all_box_coord_marginals, slopes, slopes_h, slopes_marginals, cont_page, polygons_lines_xml, ocr_all_textlines)
                        self.logger.info("Job done in %.1fs", time.time() - t0)
                        if not self.dir_in:
                            return pcgts
                        
                        
                    else:
                        contours_only_text_parent_h = None
                        if self.reading_order_machine_based:
                            order_text_new, id_of_texts_tot = self.do_order_of_regions_with_machine_optimized_algorithm(contours_only_text_parent, contours_only_text_parent_h, text_regions_p)
                        else:
                            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                                order_text_new, id_of_texts_tot = self.do_order_of_regions(contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot)
                            else:
                                contours_only_text_parent_d_ordered = self.return_list_of_contours_with_desired_order(contours_only_text_parent_d_ordered, index_by_text_par_con)
                                #try:
                                    #contours_only_text_parent_d_ordered = list(np.array(contours_only_text_parent_d_ordered, dtype=object)[index_by_text_par_con])
                                #except:
                                    #contours_only_text_parent_d_ordered = list(np.array(contours_only_text_parent_d_ordered, dtype=np.int32)[index_by_text_par_con])
                                order_text_new, id_of_texts_tot = self.do_order_of_regions(contours_only_text_parent_d_ordered, contours_only_text_parent_h, boxes_d, textline_mask_tot_d)
                            

                        if self.ocr:

                            device = cuda.get_current_device()
                            device.reset()
                            gc.collect()
                            model_ocr = VisionEncoderDecoderModel.from_pretrained(self.model_ocr_dir)
                            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
                            torch.cuda.empty_cache()
                            model_ocr.to(device)
                            
                            ind_tot = 0
                            #cv2.imwrite('./img_out.png', image_page)
                            
                            ocr_all_textlines = []
                            for indexing, ind_poly_first in enumerate(all_found_textline_polygons):
                                ocr_textline_in_textregion = []
                                for indexing2, ind_poly in enumerate(ind_poly_first):
                                    if not (self.textline_light or self.curved_line):
                                        ind_poly = copy.deepcopy(ind_poly)
                                        box_ind = all_box_coord[indexing]
                                        #print(ind_poly,np.shape(ind_poly), 'ind_poly')
                                        #print(box_ind)
                                        ind_poly = self.return_textline_contour_with_added_box_coordinate(ind_poly, box_ind)
                                        #print(ind_poly_copy)
                                        ind_poly[ind_poly<0] = 0
                                    x, y, w, h = cv2.boundingRect(ind_poly)
                                    #print(ind_poly_copy, np.shape(ind_poly_copy))
                                    #print(x, y, w, h, h/float(w),'ratio')
                                    h2w_ratio = h/float(w)
                                    mask_poly = np.zeros(image_page.shape)
                                    if not self.light_version:
                                        img_poly_on_img = np.copy(image_page)
                                    else:
                                        img_poly_on_img = np.copy(img_bin_light)

                                    mask_poly = cv2.fillPoly(mask_poly, pts=[ind_poly], color=(1, 1, 1))
                                    
                                    if self.textline_light:
                                        mask_poly = cv2.dilate(mask_poly, KERNEL, iterations=1)
                                    
                                    img_poly_on_img[:,:,0][mask_poly[:,:,0] ==0] = 255
                                    img_poly_on_img[:,:,1][mask_poly[:,:,0] ==0] = 255
                                    img_poly_on_img[:,:,2][mask_poly[:,:,0] ==0] = 255
                                    
                                    img_croped = img_poly_on_img[y:y+h, x:x+w, :]
                                    text_ocr = self.return_ocr_of_textline_without_common_section(img_croped, model_ocr, processor, device, w, h2w_ratio, ind_tot)
                                    
                                    ocr_textline_in_textregion.append(text_ocr)
                                
                                    ##cv2.imwrite(str(ind_tot)+'.png', img_croped)
                                    ind_tot = ind_tot +1
                                ocr_all_textlines.append(ocr_textline_in_textregion)
                                
                        else:
                            ocr_all_textlines = None
                        #print(ocr_all_textlines)
                        self.logger.info("detection of reading order took %.1fs", time.time() - t_order)
                        pcgts = self.writer.build_pagexml_no_full_layout(txt_con_org, page_coord, order_text_new, id_of_texts_tot, all_found_textline_polygons, all_box_coord, polygons_of_images, polygons_of_marginals, all_found_textline_polygons_marginals, all_box_coord_marginals, slopes, slopes_marginals, cont_page, polygons_lines_xml, contours_tables, ocr_all_textlines)
                        self.logger.info("Job done in %.1fs", time.time() - t0)
                        if not self.dir_in:
                            return pcgts
                    #print("text region early 7 in %.1fs", time.time() - t0)
                else:
                    _ ,_, _, textline_mask_tot_ea, img_bin_light = self.get_regions_light_v(img_res, is_image_enhanced, num_col_classifier, skip_layout_and_reading_order=self.skip_layout_and_reading_order)
                    
                    page_coord, image_page, textline_mask_tot_ea, img_bin_light, cont_page = self.run_graphics_and_columns_without_layout(textline_mask_tot_ea, img_bin_light)
                    
                    
                    ##all_found_textline_polygons =self.scale_contours_new(textline_mask_tot_ea)
                    
                    cnt_clean_rot_raw, hir_on_cnt_clean_rot = return_contours_of_image(textline_mask_tot_ea)
                    all_found_textline_polygons = filter_contours_area_of_image(textline_mask_tot_ea, cnt_clean_rot_raw, hir_on_cnt_clean_rot, max_area=1, min_area=0.00001)
                    
                    all_found_textline_polygons=[ all_found_textline_polygons ]
                    
                    all_found_textline_polygons = self.dilate_textregions_contours_textline_version(all_found_textline_polygons)
                    all_found_textline_polygons = self.filter_contours_inside_a_bigger_one(all_found_textline_polygons, textline_mask_tot_ea, type_contour="textline")
                    
                    
                    order_text_new = [0]
                    slopes =[0]
                    id_of_texts_tot =['region_0001']
                    
                    polygons_of_images = []
                    slopes_marginals = []
                    polygons_of_marginals = []
                    all_found_textline_polygons_marginals = []
                    all_box_coord_marginals = []
                    polygons_lines_xml = []
                    contours_tables = []
                    ocr_all_textlines = None
                    
                    pcgts = self.writer.build_pagexml_no_full_layout(cont_page, page_coord, order_text_new, id_of_texts_tot, all_found_textline_polygons, page_coord, polygons_of_images, polygons_of_marginals, all_found_textline_polygons_marginals, all_box_coord_marginals, slopes, slopes_marginals, cont_page, polygons_lines_xml, contours_tables, ocr_all_textlines)
                    if not self.dir_in:
                        return pcgts
                
                if self.dir_in:
                    self.writer.write_pagexml(pcgts)
                #self.logger.info("Job done in %.1fs", time.time() - t0)
                print("Job done in %.1fs", time.time() - t0)
            
        if self.dir_in:
            self.logger.info("All jobs done in %.1fs", time.time() - t0_tot)