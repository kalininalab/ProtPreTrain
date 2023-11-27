#!/bin/bash

# Loop indefinitely
while true; do
    # Run the wandb sync command
    wandb sync --sync-all
    wandb sync --clean-old-hours 240
    echo "Sleeping"

    # Wait for 60 seconds
    sleep 60
done
