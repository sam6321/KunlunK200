#!/usr/bin/env bash
# Card-less install. Downloads the public XRE and DDK archives, then compiles.
exec bash "$(cd "$(dirname "$0")" && pwd)/ci.sh"
