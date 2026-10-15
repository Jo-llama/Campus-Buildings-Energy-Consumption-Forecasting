#!/usr/bin/env bash
# Manage the Azure training VM from your own machine (Git Bash / Linux / macOS).
# Needs the Azure CLI and a prior `az login`.
#
#   scripts/azure_vm.sh create      # resource group + Ubuntu VM
#   scripts/azure_vm.sh train       # copy vm_train.sh to the VM and run it (detached, survives SSH drops)
#   scripts/azure_vm.sh logs        # follow the training log
#   scripts/azure_vm.sh fetch       # copy results/ and models/ back here
#   scripts/azure_vm.sh ssh         # open a shell on the VM
#   scripts/azure_vm.sh stop        # deallocate: compute billing stops, disk is kept
#   scripts/azure_vm.sh start       # start a deallocated VM again
#   scripts/azure_vm.sh delete      # delete the whole resource group (VM, disk, IP)
#
# Override any default with env vars, e.g. VM_SIZE=Standard_D4s_v7 scripts/azure_vm.sh create
set -euo pipefail

RG="${RG:-campus-energy-rg}"
VM="${VM:-campus-energy-vm}"
# westeurope is not accepting new customers (RequestDisallowedByAzure)
LOCATION="${LOCATION:-swedencentral}"
# 4 vCPU / 32 GB fits the free-trial quota of 4 vCPUs per region. Check which
# sizes your subscription may use with: az vm list-skus -l <region> --size Standard_E4
VM_SIZE="${VM_SIZE:-Standard_E4s_v5}"
# v6/v7 sizes only support NVMe disks: use DISK_CTRL=NVMe with those.
DISK_CTRL="${DISK_CTRL:-SCSI}"
ADMIN="${ADMIN:-azureuser}"
HORIZONS="${HORIZONS:-1 24}"
SHUTDOWN_UTC="${SHUTDOWN_UTC:-0500}"
# Spot VMs cost much less but can be evicted; set SPOT=1 to use one.
SPOT="${SPOT:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Sets $HOST once per run (`az vm show -d` is slow and occasionally fails transiently)
resolve_host() {
  local ip
  ip="$(az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv | tr -d '\r')"
  [[ -n "$ip" ]] || { echo "Could not get the VM's public IP - is it running? ($0 start)" >&2; exit 1; }
  HOST="$ADMIN@$ip"
}
remote() { ssh -o StrictHostKeyChecking=accept-new "$HOST" "$@"; }

case "${1:-}" in
  create)
    az group create -n "$RG" -l "$LOCATION" -o none
    spot_args=()
    [[ "$SPOT" == "1" ]] && spot_args=(--priority Spot --eviction-policy Deallocate --max-price -1)
    az vm create -g "$RG" -n "$VM" \
      --image Ubuntu2204 --size "$VM_SIZE" \
      --admin-username "$ADMIN" --generate-ssh-keys \
      --os-disk-size-gb 64 --public-ip-sku Standard \
      --disk-controller-type "$DISK_CTRL" \
      "${spot_args[@]}" -o table
    # Safety net in case you forget `stop`: daily shutdown at SHUTDOWN_UTC (HHMM).
    # Make sure it doesn't fall in the middle of a training run.
    az vm auto-shutdown -g "$RG" -n "$VM" --time "$SHUTDOWN_UTC" -o none
    resolve_host
    echo "VM ready at ${HOST#*@}. Next: $0 train"
    ;;
  train)
    resolve_host
    scp -o StrictHostKeyChecking=accept-new "$ROOT/scripts/vm_train.sh" "$HOST:~/vm_train.sh"
    # Pass Kaggle credentials through if set locally (needed only if the download asks for them)
    remote "KAGGLE_USERNAME='${KAGGLE_USERNAME:-}' KAGGLE_KEY='${KAGGLE_KEY:-}' HORIZONS='$HORIZONS' \
            nohup bash ~/vm_train.sh > ~/train.log 2>&1 < /dev/null &"
    echo "Training started. Follow it with: $0 logs"
    ;;
  logs)
    resolve_host
    remote "tail -n 50 -f ~/train.log"
    ;;
  fetch)
    dest="$ROOT/azure_out"
    mkdir -p "$dest"
    resolve_host
    scp -r "$HOST:~/project/results" "$HOST:~/project/models" "$HOST:~/train.log" "$dest/"
    echo "Copied to $dest"
    ;;
  ssh)
    resolve_host
    remote
    ;;
  stop)
    az vm deallocate -g "$RG" -n "$VM"
    echo "Deallocated: you're no longer paying for compute (the disk costs a little until you delete)."
    ;;
  start)
    az vm start -g "$RG" -n "$VM"
    ;;
  delete)
    az group delete -n "$RG" --yes --no-wait
    echo "Deleting resource group $RG"
    ;;
  *)
    sed -n '2,15p' "$0"
    exit 1
    ;;
esac
