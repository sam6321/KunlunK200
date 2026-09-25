#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/run.sh" int8 --loops 5000 --warmup 50 "$@"
