#!/bin/sh
#
# Machine Inspectio - Real-time Edge Yocto Project Build Environment Setup Script
#
# Copyright 2021 NXP
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA


ROOTDIR=`pwd`
PROGNAME="real-time-edge-setup-env.sh"

# A single DISTRO and MACHINE is currently supported
DISTRO="nxp-real-time-edge"
MACHINE="imx8mpevk"
BUILD_DIR="build-machine-inspection-demo"

IMAGE_FILE="${ROOTDIR}/sources/meta-real-time-edge/recipes-nxp/images/nxp-image-real-time-edge.bb"
DISTRO_FILE="${ROOTDIR}/sources/meta-real-time-edge/conf/distro/include/imx-real-time-edge-env.inc"

# Change the Real Time Edge image settings to build the "Full Image"
sed -i 's/real-time-edge-IMAGE_BASE ?= "recipes-fsl\/images\/imx-image-core.bb"/real-time-edge-IMAGE_BASE = "dynamic-layers\/qt5-layer\/recipes-fsl\/images\/imx-image-full.bb"/g' $IMAGE_FILE

# Change Distro features to add X11 support
sed -i 's/DISTRO_FEATURES_remove = "directfb x11 "/DISTRO_FEATURES_remove = "directfb "/g' $DISTRO_FILE
sed -i 's/DISTRO_FEATURES_append = " wayland pam systemd"/DISTRO_FEATURES_append = " x11 wayland pam"/g' $DISTRO_FILE

# Setup Real Time Edge environment
source ./$PROGNAME

# Add demo meta-layer to bblayer.conf
echo "" >> $BUILD_DIR/conf/bblayers.conf
echo "# Machine inspection demo layer" >> $BUILD_DIR/conf/bblayers.conf
echo "BBLAYERS += \"\${BSPDIR}/sources/meta-machine-inspection-demo\"" >> $BUILD_DIR/conf/bblayers.conf
