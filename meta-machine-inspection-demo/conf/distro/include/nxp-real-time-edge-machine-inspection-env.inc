include conf/distro/include/fsl-imx-base.inc
include conf/distro/include/fsl-imx-preferred-env.inc
include conf/distro/include/real-time-edge-base.inc
include conf/distro/include/igh-ethercat.inc
include conf/distro/include/libopen62541.inc
include conf/distro/include/rtos-industrial-examples.inc

# Remove conflicting backends
DISTRO_FEATURES:remove = "directfb "
DISTRO_FEATURES:append = " wayland pam systemd"