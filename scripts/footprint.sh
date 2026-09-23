#!/bin/bash
#
# Disk and queue footprint on the Stanford SC cluster (Andrea's standing request, 2026-09-23).
#
#     bash scripts/footprint.sh            # from the laptop: asks scdt for disk, sc for the queue
#
# Disk: the size of each top-level results tree under the parcelmate checkout, the whole of
# /juice6/u/nlp/climblab/devarda, and the share's free space. Queue: every running and pending
# job of $USER with its CPUs, memory and GPUs, and the totals of what is running now and of
# what is waiting. Read-only.

WORK=/juice6/u/nlp/climblab/devarda
U=${CLUSTER_USER:-devarda}

echo "=== disk (climblab share) ==="
ssh scdt "cd $WORK/parcelmate/results 2>/dev/null && du -sh --apparent-size */ 2>/dev/null | sort -h | tail -12; \
          echo; du -sh $WORK 2>/dev/null | awk '{print \"total under devarda:\", \$1}'; \
          df -h $WORK | tail -1 | awk '{print \"share: size\", \$2, \" used\", \$3, \" free\", \$4, \" (\" \$5 \")\"}'"

echo
echo "=== queue ($U) ==="
ssh sc "squeue -u $U -h -o '%i|%j|%T|%C|%m|%b|%M|%P|%r' | awk -F'|' '
    function gb(m) { if (m ~ /T$/) return m * 1024; if (m ~ /G$/) return m + 0; if (m ~ /M$/) return m / 1024; return m + 0 }
    function ngpu(g) { n = 0; if (g ~ /gpu/) { k = split(g, a, \":\"); n = a[k] + 0; if (n == 0) n = 1 } return n }
    { printf \"%-9s %-46s %-8s cpu %3s  mem %6s  gpu %s  %s\\n\", \$1, substr(\$2, 1, 46), \$3, \$4, \$5, ngpu(\$6), \$8
      if (\$3 == \"RUNNING\") { rc += \$4; rm += gb(\$5); rg += ngpu(\$6); rn++ }
      else if (\$9 ~ /Dependency/) { dc += \$4; dm += gb(\$5); dg += ngpu(\$6); dn++ }
      else { pc += \$4; pm += gb(\$5); pg += ngpu(\$6); pn++ } }
    END { printf \"\\nrunning now:           %3d jobs, %4d CPUs, %5.0f GB, %d GPUs\\n\", rn, rc, rm, rg
          printf \"queued for resources:  %3d jobs, %4d CPUs, %5.0f GB, %d GPUs\\n\", pn, pc, pm, pg
          printf \"waiting on dependency: %3d jobs, %4d CPUs, %5.0f GB, %d GPUs (not requested yet)\\n\", dn, dc, dm, dg }'"
