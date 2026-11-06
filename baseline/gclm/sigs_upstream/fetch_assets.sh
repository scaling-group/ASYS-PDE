#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p assets
revision=b7e5f6d44f0b1dd39a64aaf4e4a3dd30d5497140
while read -r hash name; do
  target="assets/$name"
  if [[ -f "$target" ]] && printf '%s  %s\n' "$hash" "$target" | sha256sum --check --status; then
    echo "Verified $target"
    continue
  fi
  temporary=$(mktemp "assets/.$name.XXXXXX")
  trap 'rm -f "$temporary"' EXIT
  curl --fail --location --retry 2 "https://media.githubusercontent.com/media/oroikono/SIGS/$revision/data/$name" --output "$temporary"
  printf '%s  %s\n' "$hash" "$temporary" | sha256sum --check --status
  mv "$temporary" "$target"
  trap - EXIT
  echo "Verified $target"
done <<'ASSETS'
dff87971d16bab86e4b203152fde76b0d662126bdd652b3da8ef6959d5ff857c clusters.pkl
2c6834a9df30a5c189aeec073005b409f91d6596506cfaae698dacbbcfd16f7c model.ckpt
ASSETS
