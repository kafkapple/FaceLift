#!/bin/bash
# =============================================================================
# Server Resource Monitor Script
# Usage: ./scripts/check_server_resources.sh [--watch]
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Thresholds for warnings
MEM_WARN_PERCENT=80
DISK_WARN_PERCENT=85
GPU_MEM_WARN_PERCENT=90
LOAD_WARN_PER_CORE=2.0

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Server Resource Monitor - $(hostname) - $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

check_status() {
    local value=$1
    local threshold=$2
    if (( $(echo "$value >= $threshold" | bc -l) )); then
        echo -e "${RED}[!]${NC}"
    else
        echo -e "${GREEN}[OK]${NC}"
    fi
}

# =============================================================================
# GPU Status
# =============================================================================
check_gpu() {
    echo -e "\n${YELLOW}=== GPU Status ===${NC}"

    if ! command -v nvidia-smi &> /dev/null; then
        echo "nvidia-smi not found"
        return
    fi

    # Get GPU info
    gpu_info=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)

    if [ -z "$gpu_info" ]; then
        echo "No GPU found or nvidia-smi error"
        return
    fi

    echo -e "┌─────┬────────────────────┬──────────────────┬───────┬──────┬────────┐"
    echo -e "│ GPU │ Name               │ Memory           │ Util  │ Temp │ Status │"
    echo -e "├─────┼────────────────────┼──────────────────┼───────┼──────┼────────┤"

    while IFS=',' read -r idx name mem_used mem_total util temp; do
        # Trim whitespace
        idx=$(echo "$idx" | xargs)
        name=$(echo "$name" | xargs | cut -c1-18)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        util=$(echo "$util" | xargs)
        temp=$(echo "$temp" | xargs)

        mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc)
        status=$(check_status "$mem_percent" "$GPU_MEM_WARN_PERCENT")

        printf "│ %3s │ %-18s │ %5s/%5s MB   │ %4s%% │ %3s°C│ %s │\n" \
            "$idx" "$name" "$mem_used" "$mem_total" "$util" "$temp" "$status"
    done <<< "$gpu_info"

    echo -e "└─────┴────────────────────┴──────────────────┴───────┴──────┴────────┘"

    # GPU Processes
    echo -e "\n${YELLOW}=== GPU Processes ===${NC}"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader 2>/dev/null | \
    while IFS=',' read -r uuid pid mem proc; do
        pid=$(echo "$pid" | xargs)
        mem=$(echo "$mem" | xargs)
        proc=$(echo "$proc" | xargs | rev | cut -d'/' -f1 | rev | cut -c1-40)
        gpu_idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$uuid" | cut -d',' -f1 | xargs)
        printf "  GPU %s: PID %-6s %8s  %s\n" "$gpu_idx" "$pid" "$mem" "$proc"
    done

    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
        echo "  No GPU processes running"
    fi
}

# =============================================================================
# Memory Status
# =============================================================================
check_memory() {
    echo -e "\n${YELLOW}=== Memory Status ===${NC}"

    # Parse free output
    read -r total used free shared buff_cache available <<< $(free -g | awk 'NR==2 {print $2, $3, $4, $5, $6, $7}')

    used_percent=$(echo "scale=1; $used * 100 / $total" | bc)
    avail_percent=$(echo "scale=1; $available * 100 / $total" | bc)
    status=$(check_status "$used_percent" "$MEM_WARN_PERCENT")

    echo -e "┌──────────────┬──────────┬──────────┬──────────┬────────┐"
    echo -e "│ Total        │ Used     │ Available│ Buff/Cache│ Status │"
    echo -e "├──────────────┼──────────┼──────────┼──────────┼────────┤"
    printf "│ %8s GB  │ %4s GB  │ %4s GB  │ %4s GB  │   %s  │\n" \
        "$total" "$used" "$available" "$buff_cache" "$status"
    echo -e "└──────────────┴──────────┴──────────┴──────────┴────────┘"
    printf "  Usage: %.1f%% used, %.1f%% available\n" "$used_percent" "$avail_percent"

    # Top memory consumers
    echo -e "\n  Top 5 Memory Consumers:"
    ps aux --sort=-%mem | head -6 | tail -5 | awk '{printf "    %-10s %6.1f%% %6.0f MB  %s\n", $1, $4, $6/1024, $11}'
}

# =============================================================================
# CPU Status
# =============================================================================
check_cpu() {
    echo -e "\n${YELLOW}=== CPU Status ===${NC}"

    # CPU info
    cores=$(nproc)
    model=$(cat /proc/cpuinfo | grep "model name" | head -1 | cut -d':' -f2 | xargs | cut -c1-50)

    # Load average
    read -r load1 load5 load15 <<< $(cat /proc/loadavg | awk '{print $1, $2, $3}')
    load_per_core=$(echo "scale=2; $load1 / $cores" | bc)

    if (( $(echo "$load_per_core >= $LOAD_WARN_PER_CORE" | bc -l) )); then
        status="${RED}[!]${NC}"
    else
        status="${GREEN}[OK]${NC}"
    fi

    echo -e "┌──────────────────────────────────────────────────────────────┐"
    printf "│ Model: %-54s │\n" "$model"
    printf "│ Cores: %-54s │\n" "$cores"
    echo -e "├──────────────────────────────────────────────────────────────┤"
    printf "│ Load Average: %-6s %-6s %-6s (1/5/15 min)              │\n" "$load1" "$load5" "$load15"
    printf "│ Load per Core: %-6s                              Status: %s │\n" "$load_per_core" "$status"
    echo -e "└──────────────────────────────────────────────────────────────┘"
}

# =============================================================================
# Disk Status
# =============================================================================
check_disk() {
    echo -e "\n${YELLOW}=== Disk Status ===${NC}"

    echo -e "┌────────────────────────────┬──────────┬──────────┬───────┬────────┐"
    echo -e "│ Mount                      │ Size     │ Avail    │ Use%  │ Status │"
    echo -e "├────────────────────────────┼──────────┼──────────┼───────┼────────┤"

    df -h | grep -E '^/dev|^tmpfs' | grep -v 'loop\|snap' | while read -r line; do
        fs=$(echo "$line" | awk '{print $1}')
        size=$(echo "$line" | awk '{print $2}')
        avail=$(echo "$line" | awk '{print $4}')
        use_pct=$(echo "$line" | awk '{print $5}' | tr -d '%')
        mount=$(echo "$line" | awk '{print $6}' | cut -c1-26)

        status=$(check_status "$use_pct" "$DISK_WARN_PERCENT")
        printf "│ %-26s │ %8s │ %8s │ %4s%% │   %s  │\n" "$mount" "$size" "$avail" "$use_pct" "$status"
    done

    echo -e "└────────────────────────────┴──────────┴──────────┴───────┴────────┘"

    # Project directory size
    if [ -d "/home/joon/FaceLift" ]; then
        echo -e "\n  Project Sizes:"
        du -sh /home/joon/FaceLift/checkpoints 2>/dev/null | awk '{printf "    checkpoints/: %s\n", $1}'
        du -sh /home/joon/FaceLift/data_mouse 2>/dev/null | awk '{printf "    data_mouse/:  %s\n", $1}'
        du -sh /home/joon/FaceLift/outputs 2>/dev/null | awk '{printf "    outputs/:     %s\n", $1}'
    fi
}

# =============================================================================
# Training Status
# =============================================================================
check_training() {
    echo -e "\n${YELLOW}=== Training Processes ===${NC}"

    # Check for training processes (broader search)
    train_procs=$(ps aux | grep -E "train_diffusion|train_gslrm|train_mouse|accelerate.*train" | grep -v grep | awk '{print $2}')

    if [ -z "$train_procs" ]; then
        echo "  No training processes running"
        return
    fi

    echo -e "┌─────────┬──────────┬─────────────────────────────────────────────┐"
    echo -e "│ PID     │ Memory   │ Command                                     │"
    echo -e "├─────────┼──────────┼─────────────────────────────────────────────┤"

    ps aux | grep -E "train_diffusion|train_gslrm|train_mouse" | grep -v grep | \
    while read -r user pid cpu mem vsz rss tty stat start time cmd; do
        mem_gb=$(echo "scale=2; $rss / 1024 / 1024" | bc)
        cmd_short=$(echo "$cmd" | grep -oP '(train_\w+\.py.*|accelerate.*)' | cut -c1-43)
        if [ -n "$cmd_short" ]; then
            printf "│ %-7s │ %6.2f GB│ %-43s │\n" "$pid" "$mem_gb" "$cmd_short"
        fi
    done

    echo -e "└─────────┴──────────┴─────────────────────────────────────────────┘"

    # Total training memory
    total_mem=$(ps aux | grep -E "train_diffusion|train_gslrm|train_mouse" | grep -v grep | awk '{sum+=$6} END {printf "%.2f", sum/1024/1024}')
    printf "  Total Training Memory: %s GB\n" "$total_mem"

    # Latest log
    log_file="/home/joon/FaceLift/logs/train_mvdiff_6x_gpu1.log"
    if [ -f "$log_file" ]; then
        echo -e "\n  Latest MVDiffusion Progress:"
        tail -50 "$log_file" 2>/dev/null | grep -oP 'Steps:.*\d+/\d+.*' | tail -1 | sed 's/^/    /'
    fi
}

# =============================================================================
# Summary
# =============================================================================
print_summary() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Summary${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Quick status
    mem_avail=$(free -g | awk 'NR==2 {print $7}')
    mem_total=$(free -g | awk 'NR==2 {print $2}')
    gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    disk_avail=$(df -h / | awk 'NR==2 {print $4}')

    echo -e "  Memory:  ${GREEN}${mem_avail}GB available${NC} / ${mem_total}GB total"
    echo -e "  GPU 1:   ${GREEN}${gpu_free}MB free${NC}"
    echo -e "  Disk:    ${GREEN}${disk_avail} available${NC}"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
main() {
    clear
    print_header
    check_gpu
    check_memory
    check_cpu
    check_disk
    check_training
    print_summary
}

# Watch mode
if [ "$1" == "--watch" ] || [ "$1" == "-w" ]; then
    while true; do
        main
        echo -e "${YELLOW}Refreshing in 10 seconds... (Ctrl+C to exit)${NC}"
        sleep 10
    done
else
    main
fi
