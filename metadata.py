"""
The purpose of this file is to define the metadata of the app with minimal imports. 

DO NOT CHANGE the name of the file
"""
import re

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata


# DO NOT CHANGE the function name 
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification. 
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """
    
    # first set up some basic information
    metadata = AppMetadata(
        name="Parseq OCR Wrapper",
        description="This tool applies Parseq OCR to a video or image and generates text boxes and OCR results.",
        app_license="MIT",
        identifier="parseq-wrapper",
        url="https://github.com/clamsproject/app-parseqocr-wrapper",
        analyzer_version=[l.strip().split()[0].rsplit('@')[-1][:8] for l in open('requirements.txt').readlines() if re.match(r'parseq\.git', l)][0],
        analyzer_license="Apache 2.0",
    )
    metadata.add_input(DocumentTypes.VideoDocument)
    metadata.add_input(AnnotationTypes.BoundingBox, required=True, boxType='text')

    metadata.add_output(DocumentTypes.TextDocument)
    metadata.add_output(AnnotationTypes.Alignment)
    
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(appmetadata().jsonify(pretty=True))
