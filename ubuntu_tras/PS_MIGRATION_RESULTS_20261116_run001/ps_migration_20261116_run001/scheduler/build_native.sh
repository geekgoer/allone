#!/bin/bash
set -eu
TASK_SCHED_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TASK_JSON_INCLUDE=${JSON_INCLUDE_DIR:-"$TASK_SCHED_DIR/../../route_b_ps_handoff_20261116/third_party_reference/AirSim/MavLinkCom/common_utils"}
TASK_SCHED_BUILD=${BUILD_DIR:-"$TASK_SCHED_DIR/build"}
mkdir -p "$TASK_SCHED_BUILD"
${CXX:-g++} ${CXXFLAGS:-} -std=c++14 -O2 -Wall -Wextra -pthread -I "$TASK_JSON_INCLUDE" \
  -c "$TASK_SCHED_DIR/scheduler.cpp" -o "$TASK_SCHED_BUILD/scheduler.o"
${CXX:-g++} ${CXXFLAGS:-} -std=c++14 -O2 -Wall -Wextra -pthread -I "$TASK_JSON_INCLUDE" \
  -c "$TASK_SCHED_DIR/live_calendar.cpp" -o "$TASK_SCHED_BUILD/live_calendar.o"
for TASK_SCHED_NAME in parity_cli scheduler_tests scheduler_bench live_calendar_tests; do
  TASK_SCHED_EXE=$TASK_SCHED_NAME
  if [ "$TASK_SCHED_NAME" = parity_cli ]; then TASK_SCHED_EXE=scheduler_parity; fi
  ${CXX:-g++} ${CXXFLAGS:-} -std=c++14 -O2 -Wall -Wextra -pthread -I "$TASK_JSON_INCLUDE" \
    "$TASK_SCHED_DIR/$TASK_SCHED_NAME.cpp" "$TASK_SCHED_BUILD/scheduler.o" "$TASK_SCHED_BUILD/live_calendar.o" \
    ${LDFLAGS:-} -o "$TASK_SCHED_BUILD/$TASK_SCHED_EXE"
done
