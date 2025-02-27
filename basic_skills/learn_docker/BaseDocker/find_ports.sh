#!/bin/bash

find_available_port() {
    local start_port=8888  # Specify the starting port (adjust as needed)
    local end_port=65535  # Specify the ending port (adjust as needed)

    for port in $(seq "$start_port" "$end_port"); do
        (echo >/dev/tcp/localhost/"$port") &>/dev/null
        if [ $? -ne 0 ]; then
            echo "$port"
	    return
        fi
    done

    echo "No available ports found in the range $start_port-$end_port."
    exit 1
}

find_available_port
