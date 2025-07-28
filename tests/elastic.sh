#!/usr/bin/env bash
set -euo pipefail

BIN=./tests/elastic_test

SEED_ARGS="-udp:bind 6000 -ll:id 0 -ll:networks ucx -ll:networks udp"
ROOKIE_ARGS="-udp:bind 60001 -udp:seed 127.0.0.1:6000 -ll:id 1 -ll:networks ucx -ll:networks udp"
ROOKIE_ARGS1="-udp:bind 60002 -udp:seed 127.0.0.1:6000 -ll:id 2 -ll:networks ucx -ll:networks udp"
ROOKIE_ARGS2="-udp:bind 60003 -udp:seed 127.0.0.1:6000 -ll:id 3 -ll:networks ucx -ll:networks udp"

echo "Starting seed (node-id 0) …"
$BIN $SEED_ARGS  &
seed_pid=$!

sleep 4

echo "Starting rookie (node-id 1) …"
$BIN $ROOKIE_ARGS &
rookie_pid=$!

sleep 4

echo "Starting rookie (node-id 2) …"
$BIN $ROOKIE_ARGS1 &
rookie_pid1=$!

sleep 4

echo "Starting rookie (node-id 3) …"
$BIN $ROOKIE_ARGS2 &
rookie_pid2=$!

echo
echo "Seed   pid: $seed_pid"
echo "Rookie pid: $rookie_pid"
echo "Rookie pid: $rookie_pid1"
echo "Rookie pid: $rookie_pid2"
echo
echo "Press Ctrl-C to terminate both processes."

# Wait for either process to exit (or for Ctrl-C)
#wait -n $seed_pid $rookie_pid 2>/dev/null || true

# Clean up anything still running
#kill $seed_pid  $rookie_pid 2>/dev/null || true
#wait            $seed_pid  $rookie_pid 2>/dev/null || true
