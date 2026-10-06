#!/bin/bash
# NVIDIA 550.54.14 on Ubuntu 22.04 ppc64el (kernel 5.15): the module build fails with
#   modpost: GPL-incompatible module nvidia.ko uses GPL-only symbol 'rcu_read_unlock_strict'
# Ubuntu's ppc64el kernel inlines an unconditional call to rcu_read_unlock_strict() into
# rcu_read_unlock(). With CONFIG_RCU_STRICT_GRACE_PERIOD=n that function is an empty stub,
# so defining an empty copy inside the two modules is behaviour-neutral.
#
# Usage: apply_rcu_stub.sh <extracted NVIDIA-Linux-ppc64le-550.54.14 dir>
#   sh cuda_12.4.0_550.54.14_linux_ppc64le.run --extract=$HOME/setup/extract
#   sh $HOME/setup/extract/NVIDIA-Linux-ppc64le-550.54.14.run -x --target $HOME/setup/nvdrv
#   ./apply_rcu_stub.sh $HOME/setup/nvdrv
#   cd $HOME/setup/nvdrv && sudo ./nvidia-installer --silent --dkms --no-opengl-files
set -e
D=${1:?usage: $0 <nvidia driver dir>}
for f in kernel/nvidia/nv-vtophys.c kernel/nvidia-uvm/uvm_mem.c; do
  grep -q 'void rcu_read_unlock_strict' "$D/$f" && { echo "already patched: $f"; continue; }
  printf '\n/* ppc64el: see apply_rcu_stub.sh */\nvoid rcu_read_unlock_strict(void)\n{\n}\n' >> "$D/$f"
  echo "patched: $f"
done
