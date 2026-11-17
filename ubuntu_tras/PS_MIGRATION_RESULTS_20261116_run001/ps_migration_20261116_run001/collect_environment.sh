#!/bin/sh
# Read-only board snapshot. No process termination, networking changes or secrets.
set -eu
PS_OUT=${1:?new output directory required}
test ! -e "$PS_OUT"
mkdir "$PS_OUT"
uname -a > "$PS_OUT/uname.txt"
cat /proc/cmdline > "$PS_OUT/cmdline.txt"
cat /proc/cpuinfo > "$PS_OUT/cpuinfo.txt"
cat /proc/meminfo > "$PS_OUT/meminfo.txt"
cat /proc/modules > "$PS_OUT/modules.txt"
cat /proc/uptime > "$PS_OUT/uptime.txt"
if test -r /proc/sys/kernel/random/boot_id; then cat /proc/sys/kernel/random/boot_id > "$PS_OUT/boot_id.txt"; fi
cat /proc/net/udp > "$PS_OUT/udp.txt"
cat /proc/net/tcp > "$PS_OUT/tcp.txt"
mount > "$PS_OUT/mount.txt"
ls -l /dev/dpu /usr/lib/libn2cube.so* /usr/lib/libdpuaol.so* /usr/lib/libhineon.so* > "$PS_OUT/dnndk_files.txt"
sha256sum /usr/lib/libn2cube.so.4.2.0 /usr/lib/libdpuaol.so.1.0.0 /usr/lib/libhineon.so.1.2.0 > "$PS_OUT/dnndk_sha256.txt"
sha256sum /home/root/board_package_requant18_release001/pool237_runner \
  /home/root/board_package_requant18_release001/normalization.json \
  /home/root/board_package_requant18_release001/dpu_model_registry.json > "$PS_OUT/frozen_identity_sha256.txt"
if test -r /etc/os-release; then cp /etc/os-release "$PS_OUT/os-release.txt"; fi
if test -x /sbin/ip; then /sbin/ip -br addr > "$PS_OUT/ip_addr.txt"; fi
for PS_FILE in /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; do
  if test -r "$PS_FILE"; then cat "$PS_FILE" > "$PS_OUT/${PS_FILE##*/}.txt"; fi
done
