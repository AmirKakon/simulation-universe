"""Azure GPU VM deployment helper for cloud training.

Usage:
    python scripts/deploy_azure_gpu.py --resource-group simverse-rg --vm-name simverse-train

Prerequisites:
    - Azure CLI installed and logged in (`az login`)
    - Sufficient quota for GPU VMs in your subscription
"""

from __future__ import annotations

import argparse
import subprocess
import sys


CLOUD_INIT_SCRIPT = """#!/bin/bash
set -e

# Install Docker + NVIDIA Container Toolkit
apt-get update
apt-get install -y docker.io
systemctl enable docker
systemctl start docker

distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "GPU VM setup complete. Ready for SimVerse training."
"""


def create_vm(
    resource_group: str,
    vm_name: str,
    location: str = "eastus",
    vm_size: str = "Standard_NC6s_v3",
) -> None:
    """Create an Azure GPU VM for training."""
    print(f"Creating resource group '{resource_group}' in {location}...")
    subprocess.run(
        ["az", "group", "create", "--name", resource_group, "--location", location],
        check=True,
    )

    print(f"Creating GPU VM '{vm_name}' (size: {vm_size})...")
    subprocess.run(
        [
            "az", "vm", "create",
            "--resource-group", resource_group,
            "--name", vm_name,
            "--image", "Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest",
            "--size", vm_size,
            "--admin-username", "simverse",
            "--generate-ssh-keys",
            "--custom-data", "/dev/stdin",
            "--priority", "Spot",
            "--eviction-policy", "Deallocate",
            "--max-price", "-1",
        ],
        input=CLOUD_INIT_SCRIPT.encode(),
        check=True,
    )

    print(f"\nVM '{vm_name}' created. Connect with:")
    print(f"  az ssh vm --resource-group {resource_group} --name {vm_name}")
    print(f"\nThen run training:")
    print(f"  docker build --target train-gpu -t simverse-train-gpu .")
    print(f"  docker run --gpus all simverse-train-gpu --config configs/training/default.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy SimVerse training to Azure GPU VM")
    parser.add_argument("--resource-group", default="simverse-rg")
    parser.add_argument("--vm-name", default="simverse-train")
    parser.add_argument("--location", default="eastus")
    parser.add_argument("--vm-size", default="Standard_NC6s_v3", help="Azure VM size with GPU")
    args = parser.parse_args()

    create_vm(args.resource_group, args.vm_name, args.location, args.vm_size)


if __name__ == "__main__":
    main()
