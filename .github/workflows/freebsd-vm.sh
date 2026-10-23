#!/bin/bash
# Copyright 2026 Stanford University, NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# A FreeBSD VM for CI jobs on Linux runners: FreeBSD's own amd64 VM image,
# booted with QEMU (and KVM, if there is /dev/kvm), reached as root by ssh.
#
#   freebsd-vm.sh start RELEASE     boot FreeBSD RELEASE (e.g. 15.1)
#   freebsd-vm.sh push DIR VMDIR    copy directory DIR into the VM, as VMDIR
#   freebsd-vm.sh run COMMAND       run shell COMMAND in the VM (sh, as root)
#   freebsd-vm.sh pull VMFILE FILE  copy a file out of the VM
#   freebsd-vm.sh stop              shut the VM down
#
# The image is downloaded (once) into $FREEBSD_VM_IMAGES; the VM's disk,
# ssh key and console log are in $FREEBSD_VM_DIR.  It gets every CPU and
# three quarters of the memory, unless $FREEBSD_VM_CPUS / $FREEBSD_VM_MEM
# (MiB) say otherwise.  Needs qemu-system-x86, qemu-utils, genisoimage and
# openssh-client.

set -euo pipefail

VM_DIR=${FREEBSD_VM_DIR:-${RUNNER_TEMP:-/tmp}/freebsd-vm}
IMAGES=${FREEBSD_VM_IMAGES:-$VM_DIR/images}
PORT=${FREEBSD_VM_PORT:-10022}

ssh_vm() {
  ssh -i "$VM_DIR/key" -p "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 \
      root@127.0.0.1 "$@"
}

start() {
  local release=$1
  local name=FreeBSD-$release-RELEASE-amd64-BASIC-CLOUDINIT-ufs.qcow2
  mkdir -p "$VM_DIR" "$IMAGES"
  if [ ! -f "$IMAGES/$name" ]; then
    curl -fsSL --retry 3 -o "$IMAGES/$name.xz" \
         "https://download.freebsd.org/releases/VM-IMAGES/$release-RELEASE/amd64/Latest/$name.xz"
    xz -d -T0 "$IMAGES/$name.xz"
  fi
  # (The image's root file system grows to fill the disk when it boots.)
  rm -f "$VM_DIR/disk.qcow2"
  qemu-img create -q -f qcow2 -b "$IMAGES/$name" -F qcow2 "$VM_DIR/disk.qcow2" 40G

  # nuageinit (FreeBSD's cloud-init) runs the user data as a script at the
  # end of the first boot, from a disk labelled cidata.
  rm -f "$VM_DIR/key" "$VM_DIR/key.pub"
  ssh-keygen -q -t ed25519 -N "" -f "$VM_DIR/key"
  mkdir -p "$VM_DIR/seed"
  printf 'instance-id: realm-ci\nlocal-hostname: freebsd\n' > "$VM_DIR/seed/meta-data"
  cat > "$VM_DIR/seed/user-data" <<EOF
#!/bin/sh
mkdir -p /root/.ssh
chmod 700 /root/.ssh
echo '$(cat "$VM_DIR/key.pub")' > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
echo 'PermitRootLogin prohibit-password' >> /etc/ssh/sshd_config
sysrc sshd_enable=YES
service sshd restart || service sshd start
EOF
  genisoimage -quiet -output "$VM_DIR/seed.iso" -volid cidata -joliet -rock \
              "$VM_DIR/seed/user-data" "$VM_DIR/seed/meta-data"

  local cpus=${FREEBSD_VM_CPUS:-$(nproc)}
  local mem=${FREEBSD_VM_MEM:-$(awk '/MemTotal/ { print int($2 / 1024 * 3 / 4) }' /proc/meminfo)}
  local accel=(-machine q35)
  if [ -w /dev/kvm ]; then
    accel=(-machine q35,accel=kvm -cpu host)
  fi
  qemu-system-x86_64 "${accel[@]}" -smp "$cpus" -m "$mem" \
      -drive if=virtio,file="$VM_DIR/disk.qcow2",format=qcow2 \
      -drive if=virtio,file="$VM_DIR/seed.iso",format=raw,readonly=on \
      -netdev user,id=net0,hostfwd=tcp:127.0.0.1:$PORT-:22 \
      -device virtio-net-pci,netdev=net0 \
      -display none -serial file:"$VM_DIR/console.log" \
      -daemonize -pidfile "$VM_DIR/qemu.pid"

  local i
  for i in $(seq 1 120); do
    if ssh_vm -o ConnectTimeout=5 true 2>/dev/null; then
      ssh_vm 'freebsd-version; sysctl -n hw.ncpu hw.physmem; df -h /'
      return 0
    fi
    sleep 5
  done
  echo "freebsd-vm.sh: the VM didn't come up; its console:" >&2
  tail -n 50 "$VM_DIR/console.log" >&2
  return 1
}

case ${1:-} in
  start) start "$2" ;;
  push) tar -C "$2" -cf - . | ssh_vm "mkdir -p '$3' && tar -C '$3' -xf -" ;;
  run) shift; ssh_vm "$*" ;;
  pull) ssh_vm "cat '$2'" > "$3" ;;
  stop)
    ssh_vm 'shutdown -p now' 2>/dev/null || true
    if [ -f "$VM_DIR/qemu.pid" ]; then
      for i in $(seq 1 30); do
        kill -0 "$(cat "$VM_DIR/qemu.pid")" 2>/dev/null || exit 0
        sleep 2
      done
      kill "$(cat "$VM_DIR/qemu.pid")" 2>/dev/null || true
    fi
    ;;
  *)
    sed -n '/^#   freebsd-vm.sh/p' "$0" >&2
    exit 2
    ;;
esac
