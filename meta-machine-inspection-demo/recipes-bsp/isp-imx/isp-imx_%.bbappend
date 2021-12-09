FILESEXTRAPATHS_prepend := "${THISDIR}/${PN}:"

SRC_URI += "file://start_isp.sh"

do_install_prepend () {
   cp ${WORKDIR}/start_isp.sh ${WORKDIR}/${PN}-${PV}/imx/start_isp.sh
}
