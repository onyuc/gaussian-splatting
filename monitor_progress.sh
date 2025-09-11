#!/bin/bash

# Progress monitoring script for Gaussian Splatting training
# Shows real-time progress of all 4 GPU sessions

LOG_DIR="/home/onyuk/Workspace/gaussian-splatting/logs"
SESSION_NAME="gaussian_training"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to get frame count for each GPU
get_frame_count() {
    local gpu_id=$1
    local total=0
    for i in $(seq 0 299); do
        if [ $((i % 4)) -eq $gpu_id ]; then
            ((total++))
        fi
    done
    echo $total
}

# Function to show progress
show_progress() {
    clear
    echo -e "${BLUE}=== Gaussian Splatting Training Progress ===${NC}"
    echo "$(date)"
    echo ""
    
    local total_completed=0
    local total_failed=0
    local total_frames=300
    
    for gpu in {0..3}; do
        session_name="${SESSION_NAME}_gpu${gpu}"
        expected_frames=$(get_frame_count $gpu)
        
        # Check if session is running
        if tmux has-session -t "$session_name" 2>/dev/null; then
            status="${GREEN}RUNNING${NC}"
        else
            status="${RED}STOPPED${NC}"
        fi
        
        # Get completed and failed counts
        completed=0
        failed=0
        if [ -f "$LOG_DIR/gpu_${gpu}_completed.txt" ]; then
            completed=$(wc -l < "$LOG_DIR/gpu_${gpu}_completed.txt" 2>/dev/null || echo "0")
        fi
        if [ -f "$LOG_DIR/gpu_${gpu}_failed.txt" ]; then
            failed=$(wc -l < "$LOG_DIR/gpu_${gpu}_failed.txt" 2>/dev/null || echo "0")
        fi
        
        total_completed=$((total_completed + completed))
        total_failed=$((total_failed + failed))
        
        # Calculate progress percentage
        processed=$((completed + failed))
        if [ $expected_frames -gt 0 ]; then
            progress_pct=$((processed * 100 / expected_frames))
        else
            progress_pct=0
        fi
        
        echo -e "${BLUE}GPU $gpu:${NC} $status"
        echo "  Expected frames: $expected_frames"
        echo "  Progress: $processed/$expected_frames (${progress_pct}%)"
        echo -e "  Completed: ${GREEN}$completed${NC}, Failed: ${RED}$failed${NC}"
        
        # Show last activity
        if [ -f "$LOG_DIR/gpu_${gpu}.log" ]; then
            last_line=$(tail -n 1 "$LOG_DIR/gpu_${gpu}.log" 2>/dev/null)
            if [ -n "$last_line" ]; then
                timestamp=$(echo "$last_line" | grep -o '\[.*\]' | head -1)
                activity=$(echo "$last_line" | sed 's/\[.*\] GPU[0-9]: //')
                echo "  Last activity: $timestamp"
                echo "  Status: $activity"
            fi
        fi
        
        # Progress bar
        bar_length=40
        filled=$((progress_pct * bar_length / 100))
        printf "  ["
        for ((i=0; i<filled; i++)); do printf "="; done
        for ((i=filled; i<bar_length; i++)); do printf " "; done
        printf "] %d%%\n" $progress_pct
        
        echo ""
    done
    
    # Overall progress
    total_processed=$((total_completed + total_failed))
    overall_pct=$((total_processed * 100 / total_frames))
    
    echo -e "${YELLOW}=== Overall Progress ===${NC}"
    echo "Total frames: $total_frames"
    echo "Processed: $total_processed ($overall_pct%)"
    echo -e "Completed: ${GREEN}$total_completed${NC}, Failed: ${RED}$total_failed${NC}"
    
    # Overall progress bar
    bar_length=50
    filled=$((overall_pct * bar_length / 100))
    printf "Progress: ["
    for ((i=0; i<filled; i++)); do printf "="; done
    for ((i=filled; i<bar_length; i++)); do printf " "; done
    printf "] %d%%\n" $overall_pct
    
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo "  Ctrl+C: Exit monitor"
    echo "  tmux attach -t ${SESSION_NAME}_gpu<N>: Attach to GPU N session"
    echo "  tail -f $LOG_DIR/gpu_<N>.log: Follow GPU N log"
}

# Main monitoring loop
echo -e "${BLUE}Starting progress monitor...${NC}"
echo "Press Ctrl+C to exit"
echo ""

while true; do
    show_progress
    
    # Check if all sessions are done
    active_sessions=0
    for gpu in {0..3}; do
        if tmux has-session -t "${SESSION_NAME}_gpu${gpu}" 2>/dev/null; then
            ((active_sessions++))
        fi
    done
    
    if [ $active_sessions -eq 0 ]; then
        echo -e "${GREEN}All sessions completed!${NC}"
        
        # Show final summary
        echo ""
        echo -e "${BLUE}=== Final Summary ===${NC}"
        for gpu in {0..3}; do
            if [ -f "$LOG_DIR/gpu_${gpu}_summary.txt" ]; then
                echo -e "${YELLOW}GPU $gpu:${NC}"
                cat "$LOG_DIR/gpu_${gpu}_summary.txt" | sed 's/^/  /'
                echo ""
            fi
        done
        break
    fi
    
    sleep 10
done
