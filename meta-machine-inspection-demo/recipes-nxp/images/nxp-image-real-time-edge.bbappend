# Copyright 2021, 2024 NXP
# Released under the MIT license (see COPYING.MIT for the terms)

IMAGE_INSTALL:append = " \
    packagegroup-fsl-opencv-imx \
    packagegroup-fsl-tools-gpu \
    packagegroup-fsl-gstreamer1.0 \
    gtk+3 \
    tensorflow-lite-ethosu-delegate \
    ethos-u-vela \
    python3-pillow \
    python3-pyserial \
    machine-inspection \
    tf1-detection-zoo \
    uarm-python-sdk \
    "