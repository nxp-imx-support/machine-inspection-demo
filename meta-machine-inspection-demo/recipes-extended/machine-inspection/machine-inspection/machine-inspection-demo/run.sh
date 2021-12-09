#! /bin/bash

export DISPLAY=:0.0

ifconfig eth1 up
# Disable Pause frames
ethtool -A eth1 autoneg off rx off tx off

# Enable reception of packets on VLAN 2 
ip link add link eth1 type vlan id 2

# Enable threaded NAPI
echo 1 > /sys/class/net/eth1/threaded

# Disable interrupt coalescing
ethtool -C eth1 rx-usecs 16 tx-usecs 10000 tx-frames 1

# Setup scheduled traffic
tc qdisc add dev eth1 parent root handle 100 taprio \
num_tc 3 \
map 0 0 0 0 0 1 2 0 0 0 0 0 0 0 0 0 \
queues 1@0 1@1 1@2 \
base-time 000500000 \
sched-entry S 0x2 400000 \
sched-entry S 0x5 1600000 \
flags 0x2

# Setup Rx hardware classification to place tsn-app traffic in queue 1
modprobe cls_flower
tc qdisc add dev eth1 ingress
tc filter add dev eth1 parent ffff: protocol 802.1Q flower \
vlan_prio 5 \
hw_tc 1

avb.sh start

sleep 5
PYTHONPATH=/home/root/machine-inspection/uArm-Python-SDK/:$PYTHONPATH /home/root/machine-inspection/machine-inspection-demo/machine-inspection-demo.py --labels /home/root/machine-inspection/machine-inspection-demo/mscoco_label_map.pbtxt --model /home/root/machine-inspection/machine-inspection-demo/ssdlite_mobiledet_dsp_320x320_coco_2020_05_19.tflite --threshold 0.4 &

# Move processes off CPU core 2. Ignore errors for unmoveable processes.
for i in `ps aux | grep -v PID | awk '{print $2;}'`; do taskset -p b $i &> /dev/null; done
# Move workqueues off CPU core 2
for i in `find /sys/devices/virtual/workqueue -name cpumask`; do echo b > $i; done

# Fine-tune the CPU core affinities of the NAPI kthreads
taskset -p 2 `pgrep irq/61-eth1`
chrt -pf 66 `pgrep irq/61-eth1`
taskset -p 4 `pgrep napi/eth1-rx-1`
chrt -pf 61 `pgrep napi/eth1-rx-1`
taskset -p 4 `pgrep napi/eth1-tx-1`
chrt -pf 61 `pgrep napi/eth1-tx-1`
taskset -p 4 `pgrep napi/eth1-zc-1`
chrt -pf 60 `pgrep napi/eth1-zc-1`
taskset -p 2 `pgrep napi/eth1-rx-0`
chrt -pf 1 `pgrep napi/eth1-rx-0`
taskset -p 2 `pgrep napi/eth1-tx-2`
chrt -pf 1 `pgrep napi/eth1-tx-2`
taskset -p 8 `pgrep napi/eth1-tx-0`
taskset -p 8 `pgrep napi/eth1-rx-2`
taskset -p 8 `pgrep napi/eth1-rx-3`
taskset -p 8 `pgrep napi/eth1-rx-4`
taskset -p 8 `pgrep napi/eth1-zc-0`
taskset -p 8 `pgrep napi/eth1-zc-2`
taskset -p 8 `pgrep napi/eth1-zc-3`
taskset -p 8 `pgrep napi/eth1-zc-4`

