#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/run.sh" fp32 --loops 1000 --warmup 20 "$@"
