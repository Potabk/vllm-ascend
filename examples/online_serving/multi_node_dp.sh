#!/bin/bash

set -eo errexit

# this is a head or worker node
RUNNER_TYPE="$1"

run_head_node() {
    
}

run_worker_node() {

}
if [[ "$RUNNER_TYPE" == "header" ]]; then
  run_head_node
elif [[ "$RUNNER_TYPE" == "worker" ]]; then
  run_worker_node
else
  echo "Unknown runner type: $RUNNER_TYPE"
  exit 1
fi