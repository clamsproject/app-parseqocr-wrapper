import argparse
from typing import Union

# mostly likely you'll need these modules/classes
from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes

import torch
import cv2
from PIL import Image
import mmif
from strhub.data.module import SceneTextDataModule


class Parseq(ClamsApp):

    def __init__(self):
        super().__init__()

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory. 
        # When using the ``metadata.py`` leave this do-nothing "pass" method here. 
        pass

    def _annotate(self, mmif_obj: Union[str, dict, Mmif], **kwargs) -> Mmif:
        """
        :param mmif_obj: this mmif could contain images or video, with or without preannotated text boxes
        :param **kwargs:
        :return: annotated mmif as string
        """
        parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()  # todo find out where to move this
        img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
        videoObj = cv2.VideoCapture(mmif_obj.get_document_location(DocumentTypes.VideoDocument))

        new_view: mmif.View = mmif_obj.new_view()
        self.sign_view(new_view)
        new_view.new_contain(DocumentTypes.TextDocument)

        textbox_view = mmif_obj.get_all_views_contain(AnnotationTypes.BoundingBox)  # add filter for only text boxes
        boxes = textbox_view[0].annotations

        for box in boxes:
            frame_number = int(box.properties["frame"])
            videoObj.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            _, im = videoObj.read()
            if im is not None:
                im = Image.fromarray(im.astype("uint8"), 'RGB')
                top_left, bottom_right = box.properties["coordinates"][0], box.properties["coordinates"][3]
                cropped = im.crop([top_left[0], top_left[1], bottom_right[0], bottom_right[1]])
                batch = img_transform(cropped).unsqueeze(0)

                logits = parseq(batch)
                pred = logits.softmax(-1)
                label, _ = parseq.tokenizer.decode(pred)
                text_document = new_view.new_textdocument(label)
                alignment = new_view.new_annotation(AnnotationTypes.Alignment)
                alignment.add_property("target", text_document.id)
                alignment.add_property("source", box.id)

        return mmif_obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", action="store", default="5000", help="set port to listen"
    )
    parser.add_argument("--production", action="store_true", help="run gunicorn server")

    parsed_args = parser.parse_args()

    # create the app instance
    app = Parseq()

    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        http_app.run()
