import argparse
import logging
import warnings

import cv2
import mmif
import torch
from PIL import Image
from clams import ClamsApp, Restifier, AppMetadata
from mmif import DocumentTypes, AnnotationTypes
from mmif.utils import video_document_helper as vdh
from strhub.data.module import SceneTextDataModule


class ParseqOCR(ClamsApp):
    
    def __init__(self):
        super().__init__()
        # download the model if it hasn't been downloaded yet, this will store the model in local cache directory
        # TODO (krim @ 7/21/23): amend this after https://github.com/clamsproject/clams-python/issues/168 is resolved
        torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()

    def _appmetadata(self) -> AppMetadata:
        pass
    
    def _annotate(self, mmif_obj: mmif.Mmif, **kwargs) -> mmif.Mmif:
        parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()

        img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

        new_view: mmif.View = mmif_obj.new_view()
        self.sign_view(new_view, kwargs)
        new_view.new_contain(AnnotationTypes.Alignment)
        new_view.new_contain(DocumentTypes.TextDocument)

        vds = mmif_obj.get_documents_by_type(DocumentTypes.VideoDocument)
        if vds:
            videoObj = vdh.capture(vds[0])
        else:
            warnings.warn("No video document found in the input MMIF.")
            return mmif_obj

        textbox_views = mmif_obj.get_all_views_contain(AnnotationTypes.BoundingBox)  # add filter for only text boxes
        for textbox_view in textbox_views:
            for box in textbox_view.get_annotations(AnnotationTypes.BoundingBox, boxType="text"):
                frame_number = vdh.convert_timepoint(mmif_obj, box, 'frame')
                videoObj.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                _, im = videoObj.read()
                if im is not None:
                    im = Image.fromarray(im.astype("uint8"), 'RGB')
                    (top_left_x, top_left_y), _, _, (bottom_right_x, bottom_right_y) = box.get_property("coordinates")
                    cropped = im.crop([top_left_x,top_left_y, bottom_right_x, bottom_right_y])
                    batch = img_transform(cropped).unsqueeze(0)

                    logits = parseq(batch)
                    pred = logits.softmax(-1)
                    label, _ = parseq.tokenizer.decode(pred)
                    self.logger.debug(f"OCR prediction: {label}")
                    text_document = new_view.new_textdocument(' '.join(label))
                    alignment = new_view.new_annotation(AnnotationTypes.Alignment)
                    alignment.add_property("target", text_document.id)
                    alignment.add_property("source", f'{textbox_view.id}:{box.id}')

        return mmif_obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen" )
    parser.add_argument("--production", action="store_true", help="run gunicorn server")

    parsed_args = parser.parse_args()

    app = ParseqOCR()
    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
