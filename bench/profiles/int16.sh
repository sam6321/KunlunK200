#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/run.sh" int16 --loops 2000 --warmup 20 "$@"
