SUMMARY = "Machine Inspection demo"

LICENSE = "NXP-Binary-EULA & GPLv2 & BSD-3-Clause & Apache-2.0"
LIC_FILES_CHKSUM = "file://${WORKDIR}/machine-inspection-demo/EULA.txt;md5=2acb50e7549e3925e6982a7920c26fd8"

SRC_URI = " \
    file://machine-inspection-demo \
"
S = "${WORKDIR}/machine-inspection-demo"

DEPEND = "genavb-tsn"
RDEPENDS_${PN} = "bash"

do_install() {
    install -d ${D}/home/root/machine-inspection/machine-inspection-demo/
    cp -r ${WORKDIR}/machine-inspection-demo/* ${D}/home/root/machine-inspection/machine-inspection-demo/
}

FILES_${PN} = " \
    /home/root/machine-inspection/machine-inspection-demo/* \
"
