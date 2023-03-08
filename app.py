from clams import ClamsApp, Restifier, AppMetadata
from mmif import DocumentTypes, AnnotationTypes
import torch
import cv2
from PIL import Image
import mmif
from strhub.data.module import SceneTextDataModule

__version__ = 0.1

class ParseqOCR(ClamsApp):
    def _appmetadata(self) -> AppMetadata:
        metadata = AppMetadata(
            name="Parseq OCR Wrapper",
            description="This tool applies Parseq OCR to a video or"
            "image and generates text boxes and OCR results.",
            app_version=__version__,
            app_license='MIT',
            analyzer_license='apache',
            url="https://github.com/clamsproject/app-parseqocr-wrapper", 
            identifier=f"http://apps.clams.ai/parseq/{__version__}",
        )
        metadata.add_input(DocumentTypes.VideoDocument)
        metadata.add_input(AnnotationTypes.BoundingBox, required=True, boxType='text')

        metadata.add_output(DocumentTypes.TextDocument)
        metadata.add_output(AnnotationTypes.Alignment)
        
        return metadata

    
    def _annotate(self, mmif_obj: mmif.Mmif, **kwargs) -> mmif.Mmif:
        """
        :param mmif_obj: this mmif could contain images or video, with or without preannotated text boxes
        :param **kwargs:
        :return: annotated mmif as string
        """
        parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval() # todo find out where to move this
        img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
        videoObj = cv2.VideoCapture(mmif_obj.get_document_location(DocumentTypes.VideoDocument))

        new_view: mmif.View = mmif_obj.new_view()
        self.sign_view(new_view)
        new_view.new_contain(DocumentTypes.TextDocument)

        textbox_view = mmif_obj.get_all_views_contain(AnnotationTypes.BoundingBox) # add filter for only text boxes
        boxes = textbox_view[0].annotations

        for box in boxes:
            frame_number = int(box.properties["frame"])
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
    ocr_tool = ParseqOCR()
    ocr_service = Restifier(ocr_tool)
    ocr_service.run()