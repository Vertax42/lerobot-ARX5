#!/usr/bin/env bash
#
# Switch the third_party submodule remotes between the public GitHub mirror
# (default, recorded in .gitmodules) and the internal GitLab server
# (.gitmodules.gitlab).
#
# Usage:
#   scripts/submodule-remote.sh github   # public GitHub (git@github.com:Vertax42/*)
#   scripts/submodule-remote.sh gitlab   # internal GitLab (git@192.168.1.61:physical-ai/*)
#
# Then fetch/checkout the submodules:
#   git submodule update --init --recursive
#
# Notes:
#   * Only your local .git/config and each submodule's origin are changed;
#     the committed .gitmodules (GitHub) is never modified.
#   * Every submodule pin exists on BOTH remotes, so either target resolves
#     the same commits.
set -euo pipefail

target="${1:-}"
case "$target" in
  github) file=".gitmodules" ;;
  gitlab) file=".gitmodules.gitlab" ;;
  *) echo "usage: $0 [github|gitlab]" >&2; exit 1 ;;
esac

root="$(git rev-parse --show-toplevel)"
src="$root/$file"
[ -f "$src" ] || { echo "error: $file not found at repo root" >&2; exit 1; }

# Ensure submodule.<name>.url entries exist in .git/config to override.
git -C "$root" submodule init >/dev/null

# Read path/url pairs from the chosen module file and apply them.
git config -f "$src" --get-regexp '^submodule\..*\.path$' | while read -r key path; do
  name="${key#submodule.}"; name="${name%.path}"
  url="$(git config -f "$src" --get "submodule.${name}.url")"
  # superproject override (used by `git submodule update`)
  git -C "$root" config "submodule.${name}.url" "$url"
  # already-checked-out submodule: repoint its origin too
  if [ -e "$root/$path/.git" ]; then
    git -C "$root/$path" remote set-url origin "$url"
  fi
  printf '  %-30s -> %s\n' "$path" "$url"
done

echo
echo "Submodule remotes set to: $target"
echo "Next: git submodule update --init --recursive"
