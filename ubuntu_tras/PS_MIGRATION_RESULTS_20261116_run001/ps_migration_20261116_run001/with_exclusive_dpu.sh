#!/bin/sh
# Board only: stop ONLY the verified release001 UDP service, restore on exit.
set -eu
PS_LOG=$1
shift
PS_PACKAGE=/home/root/board_package_requant18_release001
test ! -e "$PS_LOG"
mkdir "$PS_LOG"
PS_OLD_PID=
PS_TEST_PID=
for PS_PROC in /proc/[0-9]*; do
  test -r "$PS_PROC/cmdline" || continue
  PS_EXE=$(readlink "$PS_PROC/exe" 2>/dev/null || true)
  test "$PS_EXE" = "$PS_PACKAGE/pool237_runner" || continue
  PS_CMD=$(tr '\000' ' ' < "$PS_PROC/cmdline")
  case "$PS_CMD" in
    './pool237_runner serve --port 5239 --inputs inputs_raw.txt --warmup 100 --out '*)
      test -z "$PS_OLD_PID" || { echo 'multiple known services; refuse'; exit 1; }
      PS_OLD_PID=${PS_PROC#/proc/}
      ;;
    *) echo 'Other release001 runner active; refuse exclusive measurement'; exit 1 ;;
  esac
done
for PS_PROC in /proc/[0-9]*; do
  for PS_FD in "$PS_PROC"/fd/*; do
    PS_TARGET=$(readlink "$PS_FD" 2>/dev/null || true)
    if test "$PS_TARGET" = /dev/dpu && test "${PS_PROC#/proc/}" != "$PS_OLD_PID"; then
      echo "Unknown DPU owner: $PS_PROC; refuse measurement"
      exit 1
    fi
  done
done
if test -n "$PS_OLD_PID"; then
  tr '\000' ' ' < "/proc/$PS_OLD_PID/cmdline" > "$PS_LOG/original_command.txt"
  readlink "/proc/$PS_OLD_PID/cwd" > "$PS_LOG/original_cwd.txt"
  sha256sum "$PS_PACKAGE/pool237_runner" > "$PS_LOG/original_binary.sha256"
fi
restore() {
  PS_EXIT=$?
  trap - EXIT
  if test -n "$PS_OLD_PID"; then
    if test -e "/proc/$PS_OLD_PID"; then
      echo 'Original service still exists; no duplicate restore'
      exit "$PS_EXIT"
    fi
    cd "$PS_PACKAGE"
    nohup ./pool237_runner serve --port 5239 --inputs inputs_raw.txt --warmup 100 \
      --out "$PS_LOG/restored_service" > "$PS_LOG/restored_service.log" 2>&1 < /dev/null &
    echo "$!" > "$PS_LOG/restored_service.pid"
    PS_RESTORED=$!
    PS_TRIES=0
    while test "$PS_TRIES" -lt 200; do
      if grep -q '^READY ' "$PS_LOG/restored_service.log"; then
        echo "RESTORED_SERVICE_READY pid=$PS_RESTORED port=5239"
        exit "$PS_EXIT"
      fi
      if ! kill -0 "$PS_RESTORED" 2>/dev/null; then break; fi
      sleep .1
      PS_TRIES=$((PS_TRIES+1))
    done
    echo 'ERROR restoring original 5239 service; inspect preserved restore log'
    exit 3
  fi
  exit "$PS_EXIT"
}
trap restore EXIT
interrupted() {
  if test -n "$PS_TEST_PID"; then
    kill -TERM "$PS_TEST_PID" 2>/dev/null || true
    wait "$PS_TEST_PID" 2>/dev/null || true
  fi
  exit 130
}
trap interrupted INT TERM HUP
if test -n "$PS_OLD_PID"; then
  kill -INT "$PS_OLD_PID"
  PS_TRIES=0
  while test -e "/proc/$PS_OLD_PID" && test "$PS_TRIES" -lt 50; do
    sleep .1
    PS_TRIES=$((PS_TRIES+1))
  done
  if test -e "/proc/$PS_OLD_PID"; then
    PS_OLD_PID=
    echo 'Known service did not exit normally; did not force kill or run experiment'
    exit 1
  fi
fi
"$@" &
PS_TEST_PID=$!
set +e
wait "$PS_TEST_PID"
PS_STATUS=$?
set -e
PS_TEST_PID=
exit "$PS_STATUS"
