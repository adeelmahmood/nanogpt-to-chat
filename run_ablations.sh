#!/usr/bin/env bash

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 run=<experiment>[:<model>] ..."
    exit 1
fi

failures=0

for arg in "$@"; do
    [[ "$arg" == run=* ]] || {
        echo "Invalid argument: $arg"
        exit 1
    }

    spec="${arg#run=}"

    if [[ "$spec" == *:* ]]; then
        experiment="${spec%%:*}"
        model="${spec#*:}"
    else
        experiment="$spec"
        model="d12"
    fi

    [[ "$model" =~ ^d(2|12|20)$ ]] || {
        echo "Invalid model: $model"
        exit 1
    }

    ./run_ablation.sh "$experiment" "$model" || failures=$((failures + 1))
done

exit "$failures"
