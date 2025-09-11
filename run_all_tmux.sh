#!/bin/bash

# Main tmux launcher for distributed Gaussian Splatting training
# Runs GPU processes in separate tmux sessions with user-specified GPUs

GAUSSIAN_DIR="/home/onyuk/Workspace/gaussian-splatting"
SESSION_NAME="gaussian_training"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Gaussian Splatting Multi-GPU Training${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to get GPU input from user
get_gpu_input() {
    echo -e "${YELLOW}Available GPUs:${NC}"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU %s: %s (Memory: %s/%s MB)\n", $1, $2, $4, $3}'
    echo ""
    
    while true; do
        echo -e "${YELLOW}Enter GPU IDs to use (comma-separated, e.g., 0,1,2 or 1,3):${NC}"
        read -p "GPUs: " gpu_input
        
        # Remove spaces and split by comma
        IFS=',' read -ra GPU_ARRAY <<< "${gpu_input// /}"
        
        # Validate GPU IDs
        valid=true
        for gpu in "${GPU_ARRAY[@]}"; do
            if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
                echo -e "${RED}Error: '$gpu' is not a valid GPU ID${NC}"
                valid=false
                break
            fi
            
            # Check if GPU exists
            if ! nvidia-smi -i "$gpu" &>/dev/null; then
                echo -e "${RED}Error: GPU $gpu not found${NC}"
                valid=false
                break
            fi
        done
        
        if [ "$valid" = true ] && [ ${#GPU_ARRAY[@]} -gt 0 ]; then
            break
        else
            echo -e "${RED}Please enter valid GPU IDs${NC}"
        fi
    done
    
    echo ""
    echo -e "${GREEN}Selected GPUs: ${GPU_ARRAY[*]}${NC}"
    echo "Processing frames 0000-0299 across ${#GPU_ARRAY[@]} GPUs"
    
    # Show frame distribution
    for i in "${!GPU_ARRAY[@]}"; do
        gpu=${GPU_ARRAY[$i]}
        frames_example=""
        for ((f=i; f<20; f+=${#GPU_ARRAY[@]})); do
            frames_example+="$(printf "%04d" $f), "
        done
        frames_example="${frames_example%, }..."
        echo "GPU $gpu: frames $frames_example"
    done
    echo ""
}

# Change to gaussian-splatting directory
cd "$GAUSSIAN_DIR"

# Make scripts executable
chmod +x train_gpu.sh

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}Error: tmux is not installed${NC}"
    echo "Please install tmux: sudo apt-get install tmux"
    exit 1
fi

# Check GPU availability
echo -e "${YELLOW}Checking GPU availability...${NC}"
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not available${NC}"
    exit 1
fi

# Get GPU input from user
get_gpu_input

# Create logs directory
mkdir -p logs

# Kill existing sessions if they exist
echo -e "${YELLOW}Cleaning up existing tmux sessions...${NC}"
for gpu in "${GPU_ARRAY[@]}"; do
    tmux kill-session -t "${SESSION_NAME}_gpu${gpu}" 2>/dev/null || true
done

# Start tmux sessions for each selected GPU
echo -e "${GREEN}Starting tmux sessions for ${#GPU_ARRAY[@]} GPUs...${NC}"
for i in "${!GPU_ARRAY[@]}"; do
    gpu=${GPU_ARRAY[$i]}
    session_name="${SESSION_NAME}_gpu${gpu}"
    echo "Starting session: $session_name"
    
    # Create new tmux session and run the training script with GPU index in array
    tmux new-session -d -s "$session_name" -c "$GAUSSIAN_DIR" \
        "bash -c 'echo \"Starting GPU $gpu training (index $i)...\"; ./train_gpu.sh $gpu $i ${#GPU_ARRAY[@]}; echo \"GPU $gpu completed. Press any key to exit.\"; read'"
    
    echo -e "${GREEN}✓${NC} GPU $gpu session started: $session_name"
done

echo ""
echo -e "${GREEN}All GPU sessions started successfully!${NC}"
echo ""
echo -e "${YELLOW}Useful tmux commands:${NC}"
echo "  tmux list-sessions                    # List all sessions"
for gpu in "${GPU_ARRAY[@]}"; do
    echo "  tmux attach -t ${SESSION_NAME}_gpu${gpu}   # Attach to GPU ${gpu} session"
done
echo "  Ctrl+B, D                            # Detach from session"
echo "  tmux kill-session -t <session_name>  # Kill a session"
echo ""
echo -e "${YELLOW}Monitoring:${NC}"
for gpu in "${GPU_ARRAY[@]}"; do
    echo "  tail -f logs/gpu_${gpu}.log               # Monitor GPU ${gpu} progress"
done
echo ""

# Function to monitor all sessions
monitor_sessions() {
    echo -e "${BLUE}Monitoring all GPU sessions...${NC}"
    echo "Press Ctrl+C to stop monitoring"
    echo ""
    
    while true; do
        clear
        echo -e "${BLUE}=== GPU Training Status ===${NC}"
        echo "$(date)"
        echo ""
        
        for gpu in "${GPU_ARRAY[@]}"; do
            session_name="${SESSION_NAME}_gpu${gpu}"
            if tmux has-session -t "$session_name" 2>/dev/null; then
                echo -e "${GREEN}GPU $gpu: RUNNING${NC} (session: $session_name)"
                
                # Show last few lines from log if exists
                if [ -f "logs/gpu_${gpu}.log" ]; then
                    echo "  Last activity: $(tail -n 1 logs/gpu_${gpu}.log 2>/dev/null | cut -d']' -f1 | tr -d '[')"
                    
                    # Count completed and failed frames
                    completed=$(wc -l < "logs/gpu_${gpu}_completed.txt" 2>/dev/null || echo "0")
                    failed=$(wc -l < "logs/gpu_${gpu}_failed.txt" 2>/dev/null || echo "0")
                    echo "  Progress: Completed=$completed, Failed=$failed"
                fi
            else
                echo -e "${RED}GPU $gpu: STOPPED${NC}"
                
                # Show final summary if available
                if [ -f "logs/gpu_${gpu}_summary.txt" ]; then
                    echo "  $(cat logs/gpu_${gpu}_summary.txt | grep -E '(Completed|Failed)' | tr '\n' ', ')"
                fi
            fi
            echo ""
        done
        
        # Check if all sessions are done
        active_sessions=0
        for gpu in "${GPU_ARRAY[@]}"; do
            if tmux has-session -t "${SESSION_NAME}_gpu${gpu}" 2>/dev/null; then
                ((active_sessions++))
            fi
        done
        
        if [ $active_sessions -eq 0 ]; then
            echo -e "${GREEN}All GPU sessions completed!${NC}"
            break
        fi
        
        echo "Active sessions: $active_sessions/${#GPU_ARRAY[@]}"
        echo "Refreshing in 30 seconds..."
        sleep 30
    done
}

# Ask user what they want to do
echo -e "${YELLOW}What would you like to do?${NC}"
echo "1) Monitor all sessions (auto-refresh every 30s)"
echo "2) Attach to a specific GPU session"
echo "3) Exit and let sessions run in background"
echo ""
read -p "Choose option (1-3): " choice

case $choice in
    1)
        monitor_sessions
        ;;
    2)
        echo "Available sessions:"
        for i in "${!GPU_ARRAY[@]}"; do
            gpu=${GPU_ARRAY[$i]}
            echo "  $i) ${SESSION_NAME}_gpu${gpu} (GPU $gpu)"
        done
        read -p "Choose session (0-$((${#GPU_ARRAY[@]}-1))): " session_choice
        if [[ "$session_choice" =~ ^[0-9]+$ ]] && [ "$session_choice" -lt "${#GPU_ARRAY[@]}" ]; then
            selected_gpu=${GPU_ARRAY[$session_choice]}
            echo "Attaching to GPU $selected_gpu session..."
            echo "Use Ctrl+B, D to detach"
            tmux attach -t "${SESSION_NAME}_gpu${selected_gpu}"
        else
            echo "Invalid choice"
        fi
        ;;
    3)
        echo -e "${GREEN}Sessions running in background${NC}"
        echo "Use 'tmux list-sessions' to see active sessions"
        ;;
    *)
        echo "Invalid choice, exiting"
        ;;
esac

echo ""
echo -e "${BLUE}Training sessions are running in background${NC}"
echo -e "${BLUE}Use the tmux commands above to monitor progress${NC}"
