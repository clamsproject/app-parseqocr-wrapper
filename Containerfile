# Use the same base image version as the clams-python python library version
FROM ghcr.io/clamsproject/clams-python-opencv4:1.0.9
# See https://github.com/orgs/clamsproject/packages?tab=packages&q=clams-python for more base images
# IF you want to automatically publish this image to the clamsproject organization, 
# 1. you should have generated this template without --no-github-actions flag
# 1. to add arm64 support, change relevant line in .github/workflows/container.yml 
#     * NOTE that a lots of software doesn't install/compile or run on arm64 architecture out of the box 
#     * make sure you locally test the compatibility of all software dependencies before using arm64 support 
# 1. use a git tag to trigger the github action. You need to use git tag to properly set app version anyway

################################################################################
# DO NOT EDIT THIS SECTION
ARG CLAMS_APP_VERSION
ENV CLAMS_APP_VERSION ${CLAMS_APP_VERSION}
################################################################################

################################################################################
# clams-python base images are based on debian distro
# install more system packages as needed using the apt manager

# as parseq was not distributed via PyPI, need to install it from source code from the specified commit
# note that by downloading a tarball using GH API, we don't need to install `git` to perform `git clone`
ADD https://github.com/baudm/parseq/archive/bc8d95cda4666d32fa53daf2ea97ff712b71e7c7.tar.gz /parseq.tar.gz
RUN tar -x -z -f /parseq.tar.gz -C /
WORKDIR /parseq-bc8d95cda4666d32fa53daf2ea97ff712b71e7c7
# this will install the parseq in `cpu` mode, which is the default
RUN pip install -r requirements/core.txt -e .[test]
################################################################################

################################################################################
# main app installation
COPY ./ /app
WORKDIR /app
# this file has only two lines; clams-python and parseq
# the first is pre-installed in the base image
# the second is manually installed in the above
# so skipping re-installing
# RUN pip3 install -r requirements.txt

# opencv-rolling package ships more recent bugfixes
RUN pip3 install opencv-python-rolling==4.*

# pre-download model 
RUN python -c "import torch; torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()"
# default command to run the CLAMS app in a production server 
CMD ["python3", "app.py", "--production"]
################################################################################
