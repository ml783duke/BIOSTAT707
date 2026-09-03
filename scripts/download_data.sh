#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="$repo_dir/data"
base_url="https://physionet.org/files/challenge-2012/1.0.0"

mkdir -p "$data_dir"

for set_name in a b c; do
  archive="$data_dir/set-${set_name}.tar.gz"
  curl --fail --location --continue-at - \
    --output "$archive" \
    "$base_url/set-${set_name}.tar.gz"
  tar -xzf "$archive" -C "$data_dir"
done

for set_name in a b c; do
  curl --fail --location \
    --output "$data_dir/Outcomes-${set_name}.txt" \
    "$base_url/Outcomes-${set_name}.txt"
done

for set_name in a b c; do
  record_count="$(find "$data_dir/set-${set_name}" -maxdepth 1 -type f -name '*.txt' | wc -l | tr -d ' ')"
  if [[ "$record_count" != "4000" ]]; then
    echo "Expected 4000 records in set-${set_name}, found $record_count" >&2
    exit 1
  fi
  echo "set-${set_name}: $record_count records"
done

echo "Challenge 2012 data downloaded to $data_dir"
