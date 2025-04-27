#!/bin/bash

# Function to update the token
update_token() {
    TOKEN=$(oidc-token hifis_config)
    dvc remote modify --local hifis token "$TOKEN"
}

# Infinite loop to update the token every second
while true; do
    update_token
    sleep 1  # Wait for 1 second before updating again
done
