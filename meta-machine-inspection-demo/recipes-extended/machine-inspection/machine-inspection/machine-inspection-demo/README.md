Machine inspection demo
=======================
 
This is a Python3 application to be run on i.MX8MP EVK boards. It
performs object detection on images captured with a Basler camera
connected through MIPI-CSI, then directs a uARM Swift Pro robot arm to
"click" on the apples it recognizes.

The standard version talks to the robot arm through USB, while the TSN
version uses a TSN network to send commands to an RT1170 that relays
them to the actual robot.
 
Code loosely inspired from TFLite and uARM SDK examples apps.
 
Object detection model taken from:
<https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>
(mobilenet ssd neural network trained on the COCO dataset, latest model
in use is the Pixel4 DSP model)
by the TensorFlow Authors (Apache-2.0)

3D Models taken from:
<https://sketchfab.com/3d-models/lowpoly-fruits-vegetables-d3be8fed96eb48be88b47bbe8d2951e1>
by Loïc Norgeot (CC-BY-4.0)
 
Requirements:
-------------
- uARM Python SDK (need NXP version for the TSN version): FIXME
- Tensorflow lite (available in i.MX Yocto builds)
- OpenCV (available in i.MX Yocto builds)
- numpy
- pyserial (for uARM Python SDK)
