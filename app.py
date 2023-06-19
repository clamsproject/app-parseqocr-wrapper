import argparse

import cv2
import mmif
import torch
from PIL import Image
from clams import ClamsApp, Restifier, AppMetadata
from mmif import DocumentTypes, AnnotationTypes
from strhub.data.module import SceneTextDataModule


class ParseqOCR(ClamsApp):
    def _appmetadata(self) -> AppMetadata:
        pass
    
    def _annotate(self, mmif_obj: mmif.Mmif, **kwargs) -> mmif.Mmif:
        parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()  # todo find out where to move this
        img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
        videoObj = cv2.VideoCapture(mmif_obj.get_document_location(DocumentTypes.VideoDocument))

        new_view: mmif.View = mmif_obj.new_view()
        self.sign_view(new_view)
        new_view.new_contain(DocumentTypes.TextDocument)

        textbox_views = mmif_obj.get_all_views_contain(AnnotationTypes.BoundingBox)  # add filter for only text boxes
        for textbox_view in textbox_views:
            timeunit = textbox_view.metadata.contains[AnnotationTypes.BoundingBox]["timeUnit"]

            for box in textbox_view.get_annotations(AnnotationTypes.BoundingBox, boxType="text"):
                if 'frame' in timeunit:
                    frame_number = int(box.properties["timePoint"])
                else:
                    if 'millisecond' in timeunit:
                        frame_number = int(box.properties["timePoint"] / 1000 * videoObj.get(cv2.CAP_PROP_FPS))
                    elif 'second' in timeunit:
                        frame_number = int(box.properties["timePoint"] * videoObj.get(cv2.CAP_PROP_FPS))
                    else:
                        raise ValueError(f"Not supported time unit: {timeunit}")
                videoObj.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                _, im = videoObj.read()
                if im is not None:
                    im = Image.fromarray(im.astype("uint8"), 'RGB')
                    top_left, bottom_right = box.properties["coordinates"][0], box.properties["coordinates"][3]
                    cropped = im.crop([top_left[0],top_left[1], bottom_right[0], bottom_right[1]])
                    batch = img_transform(cropped).unsqueeze(0)

                    logits = parseq(batch)
                    pred = logits.softmax(-1)
                    label, _ = parseq.tokenizer.decode(pred)
                    text_document = new_view.new_textdocument(label)
                    alignment = new_view.new_annotation(AnnotationTypes.Alignment)
                    alignment.add_property("target",text_document.id)
                    alignment.add_property("source",box.id)

        return mmif_obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", action="store", default="5000", help="set port to listen"
    )
    parser.add_argument("--production", action="store_true", help="run gunicorn server")

    parsed_args = parser.parse_args()

    app = ParseqOCR()
    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        http_app.run()
