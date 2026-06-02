# FIXME: fix all of those...
# pyright: reportOptionalSubscript=false

from logging import Logger, getLogger
from typing import List, Optional
from pathlib import Path
import os
import gc
import math
from dataclasses import dataclass

import cv2
from cv2.typing import MatLike
from xml.etree import ElementTree as ET
from PIL import Image, ImageDraw
import numpy as np
from ocrd_utils import polygon_from_points, xywh_from_polygon


from .eynollah import Eynollah
from .model_zoo import EynollahModelZoo
from .utils import (
    is_image_filename,
    batched,
    pairwise,
)
from .utils.font import get_font
from .utils.xml import etree_namespace_for_element_tag
from .utils.resize import resize_image
from .utils.utils_ocr import (
    break_curved_line_into_small_pieces_and_then_merge,
    fit_text_single_line,
    get_contours_and_bounding_boxes,
    get_orientation_moments,
    preprocess_and_resize_image_for_ocrcnn_model,
    return_textlines_split_if_needed,
    rotate_image_with_padding,
)

# TODO: refine typing
@dataclass
class EynollahOcrResult:
    extracted_texts_merged: List
    extracted_confs_merged: Optional[List]
    cropped_lines_region_indexer: List
    total_bb_coordinates:List

class Eynollah_ocr(Eynollah):
    def __init__(
        self,
        *,
        model_zoo: EynollahModelZoo,
        tr_ocr=False,
        batch_size: int=0,
        do_not_mask_with_textline_contour: bool=False,
        min_conf_value_of_textline_text : float=0.3,
        logger: Optional[Logger]=None,
        device: str = '',
    ):
        self.tr_ocr = tr_ocr
        # masking for OCR and GT generation, relevant for skewed lines and bounding boxes
        self.do_not_mask_with_textline_contour = do_not_mask_with_textline_contour
        self.logger = logger if logger else getLogger('eynollah.ocr')
        
        self.min_conf_value_of_textline_text = min_conf_value_of_textline_text
        self.b_s = batch_size or 2 if tr_ocr else 8

        self.model_zoo = model_zoo
        self.setup_models(device=device)

    def setup_models(self, device=''):
        if self.tr_ocr:
            self.model_zoo.load_models('trocr_processor',
                                       ('ocr', 'tr'),
                                       device=device)
        else:
            self.model_zoo.load_models('ocr',
                                       'binarization',
                                       device=device)

    @property
    def device(self):
        return self.model_zoo.get('ocr').device

    def run_trocr(
        self,
        *,
        img: MatLike,
        page_tree: ET.ElementTree,
        page_ns,
        tr_ocr_input_height_and_width,
    ) -> EynollahOcrResult:
        
        total_bb_coordinates = []
        cropped_lines = []
        cropped_lines_region_indexer = []
        cropped_lines_meging_indexing = []

        for n_region, region in enumerate(page_tree.getroot().iter('{%s}TextRegion' % page_ns)):
            for n_line, line in enumerate(region.iter('{%s}TextLine' % page_ns)):
                cropped_lines_region_indexer.append(n_region)

                coords = line.find('{%s}Coords' % page_ns)
                if coords is None:
                    self.logger.warning("region '%s' line '%s' has no Coords", region.attrib['id'], line.attrib['id'])
                    continue
                poly = np.array(polygon_from_points(coords.attrib['points'])).astype(int)
                cont = poly[:, np.newaxis]
                xywh = xywh_from_polygon(poly)
                x, y, w, h = xywh['x'], xywh['y'], xywh['w'], xywh['h']

                total_bb_coordinates.append([x, y, w, h])

                img_crop = img[y: y + h, x: x + w]
                if not self.do_not_mask_with_textline_contour:
                    mask_poly = np.zeros(img_crop.shape[:2], dtype=np.uint8)
                    mask_poly = cv2.fillPoly(mask_poly, pts=[cont - [x, y]], color=1)
                    img_crop[mask_poly == 0] = 255 # FIXME: or median color?

                if h > 0.1 * w:
                    cropped_lines.append(resize_image(img_crop,
                                                        tr_ocr_input_height_and_width,
                                                        tr_ocr_input_height_and_width)  )
                    cropped_lines_meging_indexing.append(0)
                else:
                    splited_images, _ = return_textlines_split_if_needed(img_crop, None)
                    if splited_images:
                        cropped_lines.append(resize_image(splited_images[0],
                                                            tr_ocr_input_height_and_width,
                                                            tr_ocr_input_height_and_width))
                        cropped_lines_meging_indexing.append(1)
                        cropped_lines.append(resize_image(splited_images[1],
                                                            tr_ocr_input_height_and_width,
                                                            tr_ocr_input_height_and_width))
                        cropped_lines_meging_indexing.append(-1)
                    else:
                        cropped_lines.append(img_crop)
                        cropped_lines_meging_indexing.append(0)

        extracted_texts = []
        extracted_confs = []
        self.logger.debug("processing %d lines for %d regions",
                          len(cropped_lines), len(set(cropped_lines_region_indexer)))
        for imgs in batched(cropped_lines, self.b_s):
            pixel_values = self.model_zoo.get('trocr_processor')(
                imgs, return_tensors="pt").pixel_values
            output = self.model_zoo.get('ocr').generate(
                pixel_values.to(self.device),
                # beam search instead of greedy decoding:
                num_beams=4,
                # also return probability
                output_scores=True,
                return_dict_in_generate=True)
            if output.sequences_scores is not None:
                # log-prob averaged over length
                conf = output.sequences_scores.exp().clamp(0.0, 1.0).tolist()
            else:
                conf = [1.0] * len(output.sequences)
            if conf < self.min_conf_value_of_textline_text:
                extracted_confs.extend(0)
                extracted_texts.extend("")
                continue
            text = self.model_zoo.get('trocr_processor').batch_decode(
                output.sequences,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            extracted_confs.extend(conf)
            extracted_texts.extend(text)
        del cropped_lines
        gc.collect()

        extracted_texts_merged = [extracted_texts[ind]
                                  if cropped_lines_meging_indexing[ind] == 0
                                  else extracted_texts[ind] + " " + extracted_texts[ind + 1]
                                  for ind in range(len(cropped_lines_meging_indexing))
                                  if cropped_lines_meging_indexing[ind] >= 0]
        extracted_confs_merged = [extracted_confs[ind]
                                  if cropped_lines_meging_indexing[ind] == 0
                                  else 0.5 * (extracted_confs[ind] + extracted_confs[ind + 1])
                                  for ind in range(len(cropped_lines_meging_indexing))
                                  if cropped_lines_meging_indexing[ind] >= 0]

        return EynollahOcrResult(
            extracted_texts_merged=extracted_texts_merged,
            extracted_confs_merged=extracted_confs_merged,
            cropped_lines_region_indexer=cropped_lines_region_indexer,
            total_bb_coordinates=total_bb_coordinates,
        )
        
    def run_cnn(
        self,
        *,
        img: MatLike,
        img_bin: Optional[MatLike],
        page_tree: ET.ElementTree,
        page_ns,
        image_width,
        image_height,
    ) -> EynollahOcrResult:
        
        total_bb_coordinates = []
        cropped_lines_rgb = []
        cropped_lines_bin = []
        cropped_lines_ver_index = []
        cropped_lines_region_indexer = []
        cropped_lines_meging_indexing = []

        img_rgb = img # cosmetic
        if img_bin is None:
            # run ad-hoc binarization
            self.logger.info("running binarization for ensemble input")
            img_bin = self.do_prediction(True, img, self.model_zoo.get("binarization"),
                                         n_batch_inference=5)
            img_bin = np.repeat(img_bin[:, :, np.newaxis], 3, axis=2)
            img_bin = 255 * (img_bin == 0).astype(np.uint8)

        for n_region, region in enumerate(page_tree.getroot().iter('{%s}TextRegion' % page_ns)):
            type_textregion = region.attrib.get('type', 'paragraph')
            for n_line, line in enumerate(region.iter('{%s}TextLine' % page_ns)):
                cropped_lines_region_indexer.append(n_region)

                coords = line.find('{%s}Coords' % page_ns)
                if coords is None:
                    self.logger.warning("region '%s' line '%s' has no Coords", region.attrib['id'], line.attrib['id'])
                    continue
                poly = np.array(polygon_from_points(coords.attrib['points'])).astype(int)
                cont = poly[:, np.newaxis]
                xywh = xywh_from_polygon(poly)
                x, y, w, h = xywh['x'], xywh['y'], xywh['w'], xywh['h']
                            
                angle_radians = math.atan2(h, w)
                angle_degrees = math.degrees(angle_radians)
                if type_textregion=='drop-capital':
                    angle_degrees = 0

                total_bb_coordinates.append([x, y, w, h])
                            
                w_scaled = w * image_height / float(h)

                img_crop_rgb = img_rgb[y: y + h, x: x + w]
                img_crop_bin = img_bin[y: y + h, x: x + w]

                mask_poly = np.zeros(img_crop_rgb.shape[:2], dtype=np.uint8)
                mask_poly = cv2.fillPoly(mask_poly, pts=[cont - [x, y]], color=1)
                            
                if angle_degrees > 3:
                    better_des_slope = get_orientation_moments(cont)
                    img_crop_rgb = rotate_image_with_padding(img_crop_rgb, better_des_slope)
                    img_crop_bin = rotate_image_with_padding(img_crop_bin, better_des_slope)
                    mask_poly = rotate_image_with_padding(mask_poly, better_des_slope)
                    # get new bounding box
                    x_n, y_n, w_n, h_n = get_contours_and_bounding_boxes(mask_poly)
                    img_crop_rgb = img_crop_rgb[y_n: y_n + h_n, x_n: x_n + w_n]
                    img_crop_bin = img_crop_bin[y_n: y_n + h_n, x_n: x_n + w_n]
                    mask_poly = mask_poly[y_n: y_n + h_n, x_n: x_n + w_n]
                else:
                    better_des_slope = 0

                if not self.do_not_mask_with_textline_contour:
                    img_crop_rgb[mask_poly == 0] = 255 # FIXME: or median color?
                    img_crop_bin[mask_poly == 0] = 255

                if (type_textregion !='drop-capital' and
                    mask_poly.sum() < 0.50 * mask_poly.size and
                    w_scaled > 90):

                    img_crop_rgb, img_crop_bin = \
                        break_curved_line_into_small_pieces_and_then_merge(
                            img_crop_rgb, img_crop_bin, mask_poly)

                if w_scaled < 750:#1.5*image_width:
                    img_crop_split_rgb = img_crop_split_bin = None
                else:
                    img_crop_split_rgb, img_crop_split_bin = return_textlines_split_if_needed(
                        img_crop_rgb, img_crop_bin)
                if img_crop_split_rgb:
                    cropped_lines_rgb.extend(img_crop_split_rgb)
                    cropped_lines_bin.extend(img_crop_split_bin)
                    if abs(better_des_slope) > 45:
                        cropped_lines_ver_index.append(1)
                        cropped_lines_ver_index.append(1)
                    else:
                        cropped_lines_ver_index.append(0)
                        cropped_lines_ver_index.append(0)
                    cropped_lines_meging_indexing.append(1)
                    cropped_lines_meging_indexing.append(-1)
                else:
                    cropped_lines_rgb.append(img_crop_rgb)
                    cropped_lines_bin.append(img_crop_bin)
                    if abs(better_des_slope) > 45:
                        cropped_lines_ver_index.append(1)
                    else:
                        cropped_lines_ver_index.append(0)
                    cropped_lines_meging_indexing.append(0)

        cropped_lines_rgb = [preprocess_and_resize_image_for_ocrcnn_model(img, image_height, image_width)
                             for img in cropped_lines_rgb]
        cropped_lines_bin = [preprocess_and_resize_image_for_ocrcnn_model(img, image_height, image_width)
                             for img in cropped_lines_bin]

        extracted_texts = []
        extracted_confs = []
        self.logger.debug("processing %d lines for %d regions",
                          len(cropped_lines_rgb), len(set(cropped_lines_region_indexer)))
        cropped_lines = zip(cropped_lines_rgb, cropped_lines_bin, cropped_lines_ver_index)
        for batch in batched(cropped_lines, self.b_s):
            imgs_rgb, imgs_bin, ver_index = zip(*batch)
            ver_index = np.array(ver_index)
            imgs_rgb = np.stack(imgs_rgb)
            imgs_bin = np.stack(imgs_bin)
            imgs_rgb_ver = imgs_rgb[ver_index > 0, ::-1, ::-1]
            imgs_bin_ver = imgs_bin[ver_index > 0, ::-1, ::-1]

            # inference model now yields (char-bytes, line-prob) instead of vocidx-softmax
            # (so ctc_decode and inverse StringLookup are included)
            # also, the model now expects a secondary binary input image
            preds, probs = self.model_zoo.get('ocr').predict((imgs_rgb, imgs_bin), verbose=0)
            
            if ver_index.any():
                preds_ver, probs_ver = self.model_zoo.get('ocr').predict((imgs_rgb_ver, imgs_bin_ver), verbose=0)
                flipped_ver_is_better = np.flatnonzero(probs_ver > probs[ver_index > 0])
                if len(flipped_ver_is_better):
                    self.logger.info("%d skewed lines perform better when flipped", len(flipped_ver_is_better))
                    preds[ver_index > 0][flipped_ver_is_better] = preds_ver[flipped_ver_is_better]
                    probs[ver_index > 0][flipped_ver_is_better] = probs_ver[flipped_ver_is_better]

            def nooov(x):
                return x != b'[UNK]'
            for pred, prob in zip(preds, probs):
                if prob < self.min_conf_value_of_textline_text:
                    extracted_texts.append("")
                    extracted_confs.append(0)
                else:
                    text = b''.join(
                        filter(nooov,
                               map(bytes,
                                   (filter(None, char)
                                    for char in pred.tolist())))).decode('utf-8')
                    extracted_texts.append(text)
                    extracted_confs.append(prob)
        del cropped_lines_rgb
        del cropped_lines_bin
        gc.collect()
        
        extracted_texts_merged = [extracted_texts[ind]
                                  if cropped_lines_meging_indexing[ind] == 0
                                  else extracted_texts[ind] + " " + extracted_texts[ind + 1]
                                  for ind in range(len(cropped_lines_meging_indexing))
                                  if cropped_lines_meging_indexing[ind] >= 0]
        extracted_confs_merged = [extracted_confs[ind]
                                  if cropped_lines_meging_indexing[ind] == 0
                                  else 0.5 * (extracted_confs[ind] + extracted_confs[ind + 1])
                                  for ind in range(len(cropped_lines_meging_indexing))
                                  if cropped_lines_meging_indexing[ind] >= 0]

        return EynollahOcrResult(
            extracted_texts_merged=extracted_texts_merged,
            extracted_confs_merged=extracted_confs_merged,
            cropped_lines_region_indexer=cropped_lines_region_indexer,
            total_bb_coordinates=total_bb_coordinates,
        )
        
    def write_ocr(
        self,
        *,
        result: EynollahOcrResult,
        page_tree: ET.ElementTree,
        out_file_ocr,
        page_ns,
        img,
        out_image_with_text,
    ):
        cropped_lines_region_indexer = result.cropped_lines_region_indexer
        total_bb_coordinates = result.total_bb_coordinates
        extracted_texts_merged = result.extracted_texts_merged
        extracted_confs_merged = result.extracted_confs_merged

        unique_cropped_lines_region_indexer = np.unique(cropped_lines_region_indexer)
        if out_image_with_text:
            image_text = Image.new("RGB", (img.shape[1], img.shape[0]), "white")
            draw = ImageDraw.Draw(image_text)
            font = get_font()
            
            for indexer_text, bb_ind in enumerate(total_bb_coordinates):
                x_bb = bb_ind[0]
                y_bb = bb_ind[1]
                w_bb = bb_ind[2]
                h_bb = bb_ind[3]
                
                font = fit_text_single_line(draw, extracted_texts_merged[indexer_text],
                                            font.path, w_bb, int(h_bb*0.4) )
                
                ##draw.rectangle([x_bb, y_bb, x_bb + w_bb, y_bb + h_bb], outline="red", width=2)
                
                text_bbox = draw.textbbox((0, 0), extracted_texts_merged[indexer_text], font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                text_x = x_bb + (w_bb - text_width) // 2  # Center horizontally
                text_y = y_bb + (h_bb - text_height) // 2  # Center vertically

                # Draw the text
                draw.text((text_x, text_y), extracted_texts_merged[indexer_text], fill="black", font=font)
            image_text.save(out_image_with_text)

        text_by_textregion = []
        for ind in unique_cropped_lines_region_indexer:
            ind = np.array(cropped_lines_region_indexer)==ind
            extracted_texts_merged_un = np.array(extracted_texts_merged)[ind]
            if len(extracted_texts_merged_un)>1:
                text_by_textregion_ind = ""
                next_glue = ""
                for indt in range(len(extracted_texts_merged_un)):
                    if (extracted_texts_merged_un[indt].endswith('⸗') or
                        extracted_texts_merged_un[indt].endswith('-') or
                        extracted_texts_merged_un[indt].endswith('¬')):
                        text_by_textregion_ind += next_glue + extracted_texts_merged_un[indt][:-1]
                        next_glue = ""
                    else:
                        text_by_textregion_ind += next_glue + extracted_texts_merged_un[indt]
                        next_glue = " "
                text_by_textregion.append(text_by_textregion_ind)
            else:
                text_by_textregion.append(" ".join(extracted_texts_merged_un))

        indexer = 0
        indexer_textregion = 0
        for nn in page_tree.getroot().iter(f'{{{page_ns}}}TextRegion'):
            
            is_textregion_text = False
            for childtest in nn:
                if childtest.tag.endswith("TextEquiv"):
                    is_textregion_text = True
            
            if not is_textregion_text:
                text_subelement_textregion = ET.SubElement(nn, 'TextEquiv')
                unicode_textregion = ET.SubElement(text_subelement_textregion, 'Unicode')

            
            has_textline = False
            for child_textregion in nn:
                # FIXME: should remove Word level, if it already exists
                if child_textregion.tag.endswith("TextLine"):
                    
                    is_textline_text = False
                    for childtest2 in child_textregion:
                        if childtest2.tag.endswith("TextEquiv"):
                            is_textline_text = True
                    
                    
                    if not is_textline_text:
                        text_subelement = ET.SubElement(child_textregion, 'TextEquiv')
                        if extracted_confs_merged:
                            text_subelement.set('conf', f"{extracted_confs_merged[indexer]:.2f}")
                        unicode_textline = ET.SubElement(text_subelement, 'Unicode')
                        unicode_textline.text = extracted_texts_merged[indexer]
                    else:
                        for childtest3 in child_textregion:
                            if childtest3.tag.endswith("TextEquiv"):
                                for child_uc in childtest3:
                                    if child_uc.tag.endswith("Unicode"):
                                        if extracted_confs_merged:
                                            childtest3.set('conf', f"{extracted_confs_merged[indexer]:.2f}")
                                        child_uc.text = extracted_texts_merged[indexer]
                            
                    indexer = indexer + 1
                    has_textline = True
            if has_textline:
                if is_textregion_text:
                    for child4 in nn:
                        if child4.tag.endswith("TextEquiv"):
                            for childtr_uc in child4:
                                if childtr_uc.tag.endswith("Unicode"):
                                    childtr_uc.text = text_by_textregion[indexer_textregion]
                else:
                    unicode_textregion.text = text_by_textregion[indexer_textregion]
                indexer_textregion = indexer_textregion + 1
                
        ET.register_namespace("",page_ns)
        self.logger.info("output filename: '%s'", out_file_ocr)
        page_tree.write(out_file_ocr, xml_declaration=True, method='xml', encoding="utf-8", default_namespace=None)

    def run(
        self,
        *,
        overwrite: bool = False,
        dir_in: Optional[str] = None,
        dir_in_bin: Optional[str] = None,
        image_filename: Optional[str] = None,
        dir_xmls: str,
        dir_out_image_text: Optional[str] = None,
        dir_out: str,
    ):
        """
        Run OCR.

        Args:

            dir_in_bin (str): Prediction with RGB and binarized images for selected pages, should not be the default
        """
        if dir_in:
            ls_imgs = [os.path.join(dir_in, image_filename)
                    for image_filename in filter(is_image_filename,
                                                    os.listdir(dir_in))]
        else:
            assert image_filename
            ls_imgs = [image_filename]

        for img_filename in ls_imgs:
            file_stem = Path(img_filename).stem
            page_file_in = os.path.join(dir_xmls, file_stem+'.xml')
            out_file_ocr = os.path.join(dir_out, file_stem+'.xml')
            
            if os.path.exists(out_file_ocr):
                if overwrite:
                    self.logger.warning("will overwrite existing output file '%s'", out_file_ocr)
                else:
                    self.logger.warning("will skip input for existing output file '%s'", out_file_ocr)
                    return
                
            img = cv2.imread(img_filename)

            page_tree = ET.parse(page_file_in, parser = ET.XMLParser(encoding="utf-8"))
            page_ns = etree_namespace_for_element_tag(page_tree.getroot().tag)

            out_image_with_text = None
            if dir_out_image_text:
                out_image_with_text = os.path.join(dir_out_image_text, file_stem + '.png')

            img_bin = None
            if dir_in_bin:
                img_bin = cv2.imread(os.path.join(dir_in_bin, file_stem+'.png'))


            if self.tr_ocr:
                result = self.run_trocr(
                    img=img,
                    page_tree=page_tree,
                    page_ns=page_ns,

                    tr_ocr_input_height_and_width = 384
                )
            else:
                result = self.run_cnn( 
                    img=img,
                    page_tree=page_tree,
                    page_ns=page_ns,

                    img_bin=img_bin,
                    image_width=512,
                    image_height=32,
                )

            self.write_ocr(
                result=result,
                img=img,
                page_tree=page_tree,
                page_ns=page_ns,
                out_file_ocr=out_file_ocr,
                out_image_with_text=out_image_with_text,
            )
