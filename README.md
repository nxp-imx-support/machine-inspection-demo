# Machine Inspection Demo Application Note

This repository contains the Yocto meta-layer for the i.MX 93 as well as the MCUXpresso SDK and GenAVB patches needed to run the Machine Inspection Demo described in the _Machine Inspection Demo_ application note.

This demonstration uses the GenAVB software stack to control a robot arm on a TSN network and ensure that the system fulfills the real time requirements of the application. Indeed, the use case described in this application note simulates a conveyor belt where a camera scans the products on the belt and a robot arm selects the correct product. For this, a tablet simulates the conveyor by displaying passing oranges and apples. The i.MX 93 continuously records the tablet’s display using an onsemi AP1302 camera and processes the captured frames to determine if the passing fruits are oranges or apples. This fruit recognition is done using a TensorFlow Lite neural network running on the Neural Processing Unit of the i.MX 93. Once the coordinates of an apple are determined, the commands are sent into the TSN network composed of a i.MX RT 1180 TSN switch and a i.MX RT1170 TSN endpoint. Upon reception of the commands by the i.MX RT1170, it controls the robot arm through the UART interface in order to touch the apple. All three boards in the TSN network are configured to use Qbv settings for Quality of Service (QOS).

## License
 - License BSD-3-Clause
 - License GPL-2.0
 - License Apache-2.0
 - License MIT
 - License NXP-EULA

## Hardware and software setup
For full detaile description of the hardware and software setup, please refer to the _Machine Inspection Demo_ application note.

### Hardware
 - i.MX 93 EVK with onsemi AP1302 camera, display
 - i.MX RT1180 EVK
 - i.MX RT1170 EVK and the robot arm

### Software
 - Real Time Edge 2.7 yocto release and the additional meta-machine-inspection-demo Yocto meta-layer located in this repository
 - GenAVB/TSN MCUXpresso SDK version 4.3.0 and the apps_patches located in this repository for the i.MX RT1170
 - MCUXpresso SDK version 2.11.0 and the sdk_patches located in this repository for the i.MX RT 1170
 - GenAVB/TSN MCUXpresso SDK version 5.11.0 for the i.MX RT1180
 - MCUXpresso SDK version 2.14.2 for the i.MX RT 1170

## Build instructions
For full build instructions, please refer to the _Machine Inspection Demo_ application note.

### i.MX 93 build
The following section describes the build process,
 - Starting with the creation of a “machine-inspection-demo” folder to contain all the build elements:
```
 mkdir machine-inspection-demo
 cd machine-inspection-demo/
```
 - First, clone this repo to the machine-inspection-demo folder.

 - Then, we need to initialize the Yocto build based on the Real Time Edge v2.7 Yocto release:
```
 mkdir yocto-real-time-edge
 cd yocto-real-time-edge
 repo init -u https://github.com/real-time-edge-sw/yocto-real-time-edge.git -b real-time-edge-mickledore -m real-time-edge-2.7.0.xml
 repo sync
```
 - Before launching the build, we need to copy the machine inspection meta-layer into the source folder of the Yocto build:
```
 cp -R ../machine-inspection-demo/meta-machine-inspection-demo/ sources/
```
 - Setup the build folder:
```
DISTRO=nxp-real-time-edge-machine-inspection MACHINE=imx93evk source real-time-edge-setup-env.sh -b build-machine-inspection
```
 - Add the machine inspection layer to conf/layer.conf by adding the following line after the other layers:
```
BBLAYERS += "${BSPDIR}/sources/meta-machine-inspection-demo"
```
 - Finally, we can build the image:
```
 bitbake nxp-image-real-time-edge
```
 - Following the Real Time Edge Yocto User Guide, the final image is under <build directory>/tmp/deploy/images and can be flashed as follow:
```
 bunzip2 -dk -f <image-name>.wic.bz2
 sudo dd if=<image-name>.wic of=/dev/sd<partition> bs=1M conv=fsync
```

### RT1170 build
The image running on the i.MX RT 1170 EVK is based on the GenAVB/TSN MCUXpresso SDK version 4.3.0 and the MCUXpresso SDK version 2.11.0. The GenAVB release under the name “AVB/TSN stacks for supported i.MX RT crossover MCUs” can be download from https://www.nxp.com/design/software/development-software/mcuxpresso-software-and-tools-/wired-communications-middleware-for-nxp-microcontrollers:WIRED-COMM-MIDDLEWARE?tab=Design_Tools_Tab and the MCUXpresso SDK from https://mcuxpresso.nxp.com/.

The NXP GenAVB/TSN MCUXpresso User’s Guide under genavb_tsn-mcuxpresso-SDK_2_11_0-4_3_0/doc/ describes the needed MCU Xpresso configuration, how to setup the build environment and how to build the i.MX RT1170 TSN example application.

- In addition to the patches for GenAVB, we need to add additional patches to the MCUXpresso SDK for the machine inspection demo:
```
 cd <patch-to-GenAVB-release>/genavb_tsn-mcuxpresso-SDK_2_11_0-4_3_0/SDK_2_11_0_MIMXRT1170-EVK/
 cp <path-to-machine-inspection-demo>/sdk_patches/000* .
 patch -p0 < 0001-Don-t-use-component-IRQ-handler.patch
 patch -p0 < 0002-Adapt-driver-IRQ-handler.patch
```
 - The default configuration for the TSN application also needs to be changed: 
```
 cd <patch-to-GenAVB-release>/genavb_tsn-mcuxpresso-SDK_2_11_0-4_3_0/genavb-apps-freertos-4_3_0/
 cp <path-to-machine-inspection-demo>/apps_patches/000* .
 patch -p0 < 0001-Change-default-config-to-serial-mode.patch
```
 - Please notice that an additional patch is needed if you are using i.MX RT1170 EVK Rev B:
```
 cd <patch-to-GenAVB-release>/genavb_tsn-mcuxpresso-SDK_2_11_0-4_3_0/genavb-apps-freertos-4_3_0/
 cp <path-to-machine-inspection-demo>/apps_patches/000* .
 patch -p0 < 0002-apps-Select-Rev-B-board.patch
```

### RT1180 build
Refer to the _Machine Inspection Demo_ application note or the _NXP GenAVB/TSN MCUXpresso User’s Guide_ and
the _GenAVB/TSN Stack FreeRTOS Evaluation User's Guide_ for instructions on flashing the Bridge application
to the i.MX RT1180 EVK.

## Running the application
For detailed instructions on how to run the demo, please refer to the _Machine Inspection Demo_ application note.

On the i.MX 93 board, the application can be launched using the run.sh script:
```
 cd machine-inspection/machine-inspection-demo
 chmod a+x run.sh
 ./run.sh
```

## Known Limitations
- Support for i.MX 8MP and X forwarding has been dropped in this release.