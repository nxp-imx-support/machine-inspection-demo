SUMMARY = "Python SDK for uArm"
HOMEPAGE = "https://github.com/uarm-developer/uArm-Python-SDK"

LICENSE = "BSD"
LIC_FILES_CHKSUM = "file://LICENSE;md5=d41d8cd98f00b204e9800998ecf8427e"

SRC_URI = "git://github.com/uArm-Developer/uArm-Python-SDK.git;branch=2.0"

PV = "1.0+git${SRCPV}"
SRCREV = "936c043af842dcc6fd1ac3566fa74ab620d18286"

S = "${WORKDIR}/git"

inherit setuptools3

do_install() {
    install -d ${D}/home/root/machine-inspection/uArm-Python-SDK/
    cp -r ${S}/* ${D}/home/root/machine-inspection/uArm-Python-SDK/
}

FILES_${PN} = " \
    /home/root/machine-inspection/uArm-Python-SDK/* \
"
