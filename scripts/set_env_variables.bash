#!/usr/bin/env bash

set -x

project_name='UnsupervisedFlow'
project_short='usfl'
if [ -n "$PROJECT_SHORT" ]; then
    if [ "$PROJECT_SHORT" != "$project_short" ]; then
        set +x
        echo "There was already an activated project ($PROJECT_SHORT) in this shell. Please fire up a new shell!"
        return 1
    fi
fi
export PROJECT_NAME=$project_name
export PROJECT_SHORT=$project_short
export SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd -P)"
export CFG_DIR="$SRC_DIR/unsup_flow/config"
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export INPUT_DATADIR=/path/to/dataset
export OUTPUT_DATADIR=/path/to/output_dest

if [ -z $OUTPUT_DATADIR ]; then
    export OUTPUT_DATADIR="$DATADIR/$PROJECT_SHORT/out"
else
    if [ "/out" != ${OUTPUT_DATADIR: -4} ]; then
        export OUTPUT_DATADIR="$OUTPUT_DATADIR/$PROJECT_SHORT/out"
    fi
fi
if [ -z $INPUT_DATADIR ]; then
    export INPUT_DATADIR="$DATADIR/$PROJECT_SHORT/in"
else
    if [ "/in" != ${INPUT_DATADIR: -3} ]; then
        export INPUT_DATADIR="$INPUT_DATADIR/$PROJECT_SHORT/in"
    fi
fi

echo "SRC_DIR:           "$SRC_DIR
echo "OUTPUT_DATADIR:    "$OUTPUT_DATADIR
echo "INPUT_DATADIR:     "$INPUT_DATADIR

set +x
