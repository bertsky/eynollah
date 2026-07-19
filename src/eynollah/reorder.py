"""
Machine learning based reading order detection
"""

# pyright: reportCallIssue=false
# pyright: reportUnboundVariable=false
# pyright: reportArgumentType=false

import logging 
import os
import time
from typing import Optional
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import statistics

from ocrd_utils import (
    polygon_from_points,
    xywh_from_points,
)

from .eynollah import Eynollah
from .model_zoo import EynollahModelZoo
from .utils.resize import resize_image
from .utils.contour import (
    find_new_features_of_contours,
    return_contours_of_image,
    return_parent_contours,
)
from .utils import is_xml_filename

DPI_THRESHOLD = 298
KERNEL = np.ones((5, 5), np.uint8)


class Reorder(Eynollah):
    def __init__(
            self,
            *,
            model_zoo: EynollahModelZoo,
            logger : Optional[logging.Logger] = None,
            device: str = '',
    ):
        self.logger = logger or logging.getLogger('eynollah.mbreorder')
        self.model_zoo = model_zoo
        
        self.setup_models(device=device)

    def setup_models(self, device=''):
        loadable = ['reading_order']
        self.model_zoo.load_models(*loadable, device=device)
        for model in loadable:
            self.logger.debug("model %s has input shape %s", model,
                              self.model_zoo.get(model).input_shape)

    def read_xml(self,
                 xml_file: str,
                 label_text=1,
                 label_head=2,
                 label_imgs=5,
                 label_seps=6,
                 label_marg=8,
                 label_drop=4,
    ):
        tree1 = ET.parse(xml_file, parser = ET.XMLParser(encoding='utf-8'))
        root1=tree1.getroot()
        alltags=[elem.tag for elem in root1.iter()]
        link=alltags[0].split('}')[0]+'}'

        index_tot_regions = []
        tot_region_ref = []

        page = root1.find(link+'Page')
        height = int(page.get('imageHeight', 0))
        width = int(page.get('imageWidth', 0))

        for jj in root1.iter(link+'RegionRefIndexed'):
            index_tot_regions.append(jj.attrib['index'])
            tot_region_ref.append(jj.attrib['regionRef'])
            
        bb_coord_printspace = None
        if (link+'PrintSpace' in alltags or
            link+'Border' in alltags):
            tag_printspace = next(x for x in alltags
                                  if x.endswith(('Border', 'PrintSpace')))
            nn = page.find(tag_printspace)
            coords = nn.find(link + 'Coords')
            if points := coords.attrib.get('points'):
                xywh = xywh_from_points(points)
                bb_coord_printspace = [xywh['x'],
                                       xywh['y'],
                                       xywh['w'],
                                       xywh['h']]

        seps_cont = []
        imgs_cont = []
        text_para_cont = []
        text_para_ids = []
        text_drop_cont = []
        text_drop_ids = []
        text_head_cont = []
        text_head_ids = []
        text_marg_cont = []
        text_marg_ids = []

        for nn in root1.iter():
            if not nn.tag.endswith('Region'):
                continue
            if (coords := nn.find(link + 'Coords')) is None:
                continue
            if (points := coords.attrib.get('points')) is None:
                continue
            if (id_ := nn.get('id')) is None:
                continue
            poly = polygon_from_points(points)
            cont = np.array(poly, dtype=int)[:, np.newaxis]
            if nn.tag.endswith('}TextRegion'):
                type_ = nn.get('type', '')
                if type_ == 'drop-capital':
                    text_drop_cont.append(cont)
                    text_drop_ids.append(id_)
                elif type_ == 'heading':
                    text_head_cont.append(cont)
                    text_head_ids.append(id_)
                elif type_ == 'header':
                    # FIXME: do not keep that mapping
                    text_head_cont.append(cont)
                    text_head_ids.append(id_)
                elif type_ == 'marginalia':
                    text_marg_cont.append(cont)
                    text_marg_ids.append(id_)
                else:
                    text_para_cont.append(cont)
                    text_para_ids.append(id_)

            elif nn.tag.endswith('}GraphicRegion'):
                imgs_cont.append(cont)

            elif nn.tag.endswith('}ImageRegion'):
                imgs_cont.append(cont)

            elif nn.tag.endswith('}SeparatorRegion'):
                seps_cont.append(cont)

        img = np.zeros((height, width), dtype=np.uint8)
        img = cv2.fillPoly(img, pts=text_para_cont, color=label_text)
        img = cv2.fillPoly(img, pts=text_head_cont, color=label_head)
        img = cv2.fillPoly(img, pts=text_marg_cont, color=label_marg)
        img = cv2.fillPoly(img, pts=text_drop_cont, color=label_drop)
        img = cv2.fillPoly(img, pts=imgs_cont, color=label_imgs)
        img = cv2.fillPoly(img, pts=seps_cont, color=label_seps)

        return (tree1, root1,
                bb_coord_printspace,
                text_para_ids, text_head_ids, text_drop_ids,
                text_para_cont, text_head_cont, text_drop_cont,
                tot_region_ref,
                width, height,
                index_tot_regions,
                img)

    def run(self,
            overwrite: bool = False,
            xml_filename: Optional[str] = None,
            dir_in: Optional[str] = None,
            dir_out: Optional[str] = None,
    ):
        """
        Get image and scales, then extract the page of scanned image
        """
        self.logger.debug("enter run")

        if dir_in:
            t0_tot = time.time()
            ls_xmls  = [os.path.join(dir_in, xml_filename)
                        for xml_filename in filter(is_xml_filename,
                                                   os.listdir(dir_in))]
        elif xml_filename:
            ls_xmls = [xml_filename]
        else:
            raise ValueError("run requires either a single image filename or a directory")

        for xml_filename in ls_xmls:
            self.run_single(xml_filename, dir_out=dir_out, overwrite=overwrite)

        if dir_in:
            self.logger.info("All jobs done in %.1fs", time.time() - t0_tot)

    def run_single(self,
                   xml_filename: str,
                   dir_out: Optional[str] = None,
                   overwrite: bool = False
    ) -> None:
        self.logger.info(xml_filename)
        t0 = time.time()

        file_name = Path(xml_filename).stem
        (tree_xml, root_xml,
         _, # FIXME: crop img_poly and contours (bb_coord_printspace)
         para_ids, head_ids, drop_ids,
         para_cont, head_cont, drop_cont,
         _, # FIXME: do not ignore existing RO (tot_region_ref)
         width, height,
         _, # FIXME: do not ignore existing RO (index_tot_regions)
         region_labels) = self.read_xml(xml_filename)

        all_text_ids = np.array(para_ids + head_ids + drop_ids)

        self.logger.debug("ordering %d paragraphs, %d headings and %d drop-capitals",
                          len(para_ids), len(head_ids), len(drop_ids))
        order_text = self.do_order_of_regions_with_model(
            para_cont,
            head_cont,
            drop_cont,
            region_labels)

        all_text_ids = all_text_ids[order_text]

        alltags=[elem.tag for elem in root_xml.iter()]

        link=alltags[0].split('}')[0]+'}'
        ET.register_namespace("", link[1:-1])

        page = root_xml.find(link+'Page')
        ro_old = page.find(link+'ReadingOrder')
        if ro_old is not None:
            page.remove(ro_old)

        ro_new = ET.Element('ReadingOrder')
        ro_group = ET.SubElement(ro_new, 'OrderedGroup')
        ro_group.set('id', "ro357564684568544579089")

        for index, id_text in enumerate(all_text_ids):
            ro_ref = ET.SubElement(ro_group, 'RegionRefIndexed')
            ro_ref.set('regionRef', id_text)
            ro_ref.set('index', str(index))

        pos = len(page.findall(link+'AlternativeImage') +
                  page.findall(link+'Border') +
                  page.findall(link+'PrintSpace'))
        page.insert(pos, ro_new)

        tree_xml.write(os.path.join(dir_out or "", file_name + '.xml'),
                       xml_declaration=True,
                       method='xml',
                       encoding="utf-8",
                       default_namespace=None)
        self.logger.info("Job done in %.1fs", time.time() - t0)
            
