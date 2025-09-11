#!/bin/bash

# Main tmux launcher for distributed Gaussian Splatting training
# Runs 4 GPU processes in separate tmux sessions

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
echo "Processing frames 0000-0299 across 4 GPUs"
echo "GPU 0: frames 0000, 0004, 0008, 0012, ..."
echo "GPU 1: frames 0001, 0005, 0009, 0013, ..."
echo "GPU 2: frames 0002, 0006, 0010, 0014, ..."
echo "GPU 3: frames 0003, 0007, 0011, 0015, ..."
echo ""

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

nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
echo ""

# Create logs directory
mkdir -p logs

# Kill existing sessions if they exist
echo -e "${YELLOW}Cleaning up existing tmux sessions...${NC}"
for gpu in {0..3}; do
    tmux kill-session -t "${SESSION_NAME}_gpu${gpu}" 2>/dev/null || true
done

# Start tmux sessions for each GPU
echo -e "${GREEN}Starting tmux sessions for 4 GPUs...${NC}"
for gpu in {0..3}; do
    session_name="${SESSION_NAME}_gpu${gpu}"
    echo "Starting session: $session_name"
    
    # Create new tmux session and run the training script
    tmux new-session -d -s "$session_name" -c "$GAUSSIAN_DIR" \
        "bash -c 'echo \"Starting GPU $gpu training...\"; ./train_gpu.sh $gpu; echo \"GPU $gpu completed. Press any key to exit.\"; read'"
    
    echo -e "${GREEN}✓${NC} GPU $gpu session started: $session_name"
done

echo ""
echo -e "${GREEN}All GPU sessions started successfully!${NC}"
echo ""
echo -e "${YELLOW}Useful tmux commands:${NC}"
echo "  tmux list-sessions                    # List all sessions"
echo "  tmux attach -t ${SESSION_NAME}_gpu0   # Attach to GPU 0 session"
echo "  tmux attach -t ${SESSION_NAME}_gpu1   # Attach to GPU 1 session"
echo "  tmux attach -t ${SESSION_NAME}_gpu2   # Attach to GPU 2 session"
echo "  tmux attach -t ${SESSION_NAME}_gpu3   # Attach to GPU 3 session"
echo "  Ctrl+B, D                            # Detach from session"
echo "  tmux kill-session -t <session_name>  # Kill a session"
echo ""
echo -e "${YELLOW}Monitoring:${NC}"
echo "  tail -f logs/gpu_0.log               # Monitor GPU 0 progress"
echo "  tail -f logs/gpu_1.log               # Monitor GPU 1 progress"
echo "  tail -f logs/gpu_2.log               # Monitor GPU 2 progress"
echo "  tail -f logs/gpu_3.log               # Monitor GPU 3 progress"
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
        
        for gpu in {0..3}; do
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
        for gpu in {0..3}; do
            if tmux has-session -t "${SESSION_NAME}_gpu${gpu}" 2>/dev/null; then
                ((active_sessions++))
            fi
        done
        
        if [ $active_sessions -eq 0 ]; then
            echo -e "${GREEN}All GPU sessions completed!${NC}"
            break
        fi
        
        echo "Active sessions: $active_sessions/4"
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
        for gpu in {0..3}; do
            echo "  $gpu) ${SESSION_NAME}_gpu${gpu}"
        done
        read -p "Choose GPU (0-3): " gpu_choice
        if [[ "$gpu_choice" =~ ^[0-3]$ ]]; then
            echo "Attaching to GPU $gpu_choice session..."
            echo "Use Ctrl+B, D to detach"
            tmux attach -t "${SESSION_NAME}_gpu${gpu_choice}"
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
