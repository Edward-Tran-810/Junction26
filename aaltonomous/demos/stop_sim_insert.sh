#!/bin/bash
# Stop ONLY sim instance d (SIGINT to its python pid, matched by its AALTO_INSTANCE=d environment).
for p in $(pgrep -f "^python .*aalto_sim_main"); do
  if tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "AALTO_INSTANCE=d"; then
    kill -INT $p; for i in $(seq 120); do kill -0 $p 2>/dev/null || break; sleep 0.5; done
    kill -0 $p 2>/dev/null && echo "pid $p still running" || echo "instance d (pid $p) stopped"
  fi
done
