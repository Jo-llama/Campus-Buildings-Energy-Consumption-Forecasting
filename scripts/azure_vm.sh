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
# Override any default with env vars, e.g. VM_SIZE=Standard_E4s_v5 scripts/azure_vm.sh create
set -euo pipefail

RG="${RG:-campus-energy-rg}"
VM="${VM:-campus-energy-vm}"
LOCATION="${LOCATION:-westeurope}"
# 4 vCPU / 16 GB fits the usual free-trial quota of 4 vCPUs per region.
# If the resample step runs out of memory, use Standard_E4s_v5 (4 vCPU / 32 GB).
VM_SIZE="${VM_SIZE:-Standard_D4s_v5}"
ADMIN="${ADMIN:-azureuser}"
HORIZONS="${HORIZONS:-1 24}"
# Spot VMs cost much less but can be evicted; set SPOT=1 to use one.
SPOT="${SPOT:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ip() { az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv; }
remote() { ssh -o StrictHostKeyChecking=accept-new "$ADMIN@$(ip)" "$@"; }

case "${1:-}" in
  create)
    az group create -n "$RG" -l "$LOCATION" -o none
    spot_args=()
    [[ "$SPOT" == "1" ]] && spot_args=(--priority Spot --eviction-policy Deallocate --max-price -1)
    az vm create -g "$RG" -n "$VM" \
      --image Ubuntu2204 --size "$VM_SIZE" \
      --admin-username "$ADMIN" --generate-ssh-keys \
      --os-disk-size-gb 64 --public-ip-sku Standard \
      "${spot_args[@]}" -o table
    # Safety net: shut down every night at 23:00 UTC in case you forget `stop`.
    az vm auto-shutdown -g "$RG" -n "$VM" --time 2300 -o none
    echo "VM ready at $(ip). Next: $0 train"
    ;;
  train)
    scp -o StrictHostKeyChecking=accept-new "$ROOT/scripts/vm_train.sh" "$ADMIN@$(ip):~/vm_train.sh"
    # Pass Kaggle credentials through if set locally (needed only if the download asks for them)
    remote "KAGGLE_USERNAME='${KAGGLE_USERNAME:-}' KAGGLE_KEY='${KAGGLE_KEY:-}' HORIZONS='$HORIZONS' \
            nohup bash ~/vm_train.sh > ~/train.log 2>&1 < /dev/null &"
    echo "Training started. Follow it with: $0 logs"
    ;;
  logs)
    remote "tail -n 50 -f ~/train.log"
    ;;
  fetch)
    dest="$ROOT/azure_out"
    mkdir -p "$dest"
    scp -r "$ADMIN@$(ip):~/project/results" "$ADMIN@$(ip):~/project/models" "$ADMIN@$(ip):~/train.log" "$dest/"
    echo "Copied to $dest"
    ;;
  ssh)
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
