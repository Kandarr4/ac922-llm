#!/bin/bash
set -x
cd ~/setup
echo "[$(date)] waiting for download to finish"
while pgrep -f 'wget.*ppc64le.run' >/dev/null; do sleep 10; done
echo "[$(date)] download finished, verifying md5"
MD5=$(md5sum cuda_12.4.0_550.54.14_linux_ppc64le.run | cut -d' ' -f1)
echo "md5=$MD5"
if [ "$MD5" != "42b04366bb2d2e5b2325b493c3c96278" ]; then
  echo "MD5 MISMATCH — aborting"
  exit 1
fi
echo "[$(date)] md5 OK, starting silent install (driver + toolkit)"
sudo sh cuda_12.4.0_550.54.14_linux_ppc64le.run --silent --driver --toolkit --no-opengl-libs
echo "[$(date)] installer exit code: $?"
