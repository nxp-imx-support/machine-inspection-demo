SUMMARY = "Tensorflow model used for Machine Inspection Demo"

LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://LICENSE;md5=6f798069926aa738ee3bbbcac6c62a2f \
                    file://orbit/LICENSE;md5=175792518e4ac015ab6696d16c4f607e \
                    file://tensorflow_models/LICENSE;md5=2e2780a0da7fcd373a2f7ffb50fc24b0"

SRC_URI = "git://github.com/tensorflow/models.git;protocol=https;branch=master"
SRC_URI += "http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_dsp_320x320_coco_2020_05_19.tar.gz"
SRC_URI[sha256sum] = "1e878f540fb2f75d65b7f2d294186cf600a4e808d179af2898b076263dfb12fc"

PV = "1.0+git${SRCPV}"
SRCREV = "c8a402fc7fc0cc391a9e3ac56fb7b3ea6f9d202e"

S = "${WORKDIR}/git"

do_install () {
    MODEL_NAME=ssdlite_mobiledet_dsp_320x320_coco_2020_05_19
    MODEL_TYPE=uint8
    LABEL_GITHUB_PATH=research/object_detection/data/mscoco_label_map.pbtxt

    install -d ${D}/home/root/machine-inspection/machine-inspection-demo/
    cp ${WORKDIR}/${MODEL_NAME}/${MODEL_TYPE}/model.tflite ${D}/home/root/machine-inspection/machine-inspection-demo/${MODEL_NAME}.tflite
    cp ${S}/${LABEL_GITHUB_PATH} ${D}/home/root/machine-inspection/machine-inspection-demo/
}

FILES:${PN} = " \
    /home/root/machine-inspection/machine-inspection-demo/* \
"
