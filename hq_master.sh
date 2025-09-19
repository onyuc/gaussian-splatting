#!/bin/bash

# Master HQ training script that runs in tmux
DATASET_PATH="/mnt/HDD/onyuc/dataset/N3DV/cut_roasted_beef"
OUTPUT_PATH="/mnt/HDD/onyuc/gaussian_splatting/output/cut_roasted_beef"
MAX_ITERATIONS=10
GPU_ARRAY=(0)

GAUSSIAN_DIR="/home/onyuc/Workspace/gaussian-splatting"
cd "$GAUSSIAN_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_splatting

# Function to analyze PSNR results
analyze_psnr_results() {
    local iteration=$1
    echo -e "${CYAN}=== PSNR Analysis (Iteration $iteration) ===${NC}"
    
    # Create temporary file for PSNR values
    local psnr_file="/tmp/psnr_analysis_${iteration}.txt"
    > "$psnr_file"
    
    echo "Analyzing PSNR results from all frames..."
    
    # Collect PSNR values from all completed frames
    local total_frames=0
    local collected_frames=0
    
    for frame_num in $(seq -f "%04g" 0 299); do
        ((total_frames++))
        local model_path="$OUTPUT_PATH/F$frame_num"
        local results_file="$model_path/results.json"
        
        if [ -f "$results_file" ]; then
            # Extract PSNR value from results.json
            local psnr_value=$(python3 -c "
import json
import sys
try:
    with open('$results_file', 'r') as f:
        data = json.load(f)
    # Find the PSNR value in the nested structure
    for method_key in data:
        if 'PSNR' in data[method_key]:
            print(data[method_key]['PSNR'])
            break
except:
    sys.exit(1)
" 2>/dev/null)
            
            if [ $? -eq 0 ] && [ -n "$psnr_value" ]; then
                echo "$frame_num $psnr_value" >> "$psnr_file"
                ((collected_frames++))
            fi
        fi
    done
    
    echo "Collected PSNR data from $collected_frames/$total_frames frames"
    
    if [ $collected_frames -lt 10 ]; then
        echo -e "${RED}Error: Not enough PSNR data collected ($collected_frames frames)${NC}"
        return 1
    fi
    
    # Calculate threshold and identify poor frames
    local retrain_file="$OUTPUT_PATH/retrain_frames_iter${iteration}.txt"
    
    local retrain_count=$(python3 -c "
import sys
psnr_values = []
frame_data = []

with open('$psnr_file', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            frame_num, psnr_value = parts[0], float(parts[1])
            psnr_values.append(psnr_value)
            frame_data.append((frame_num, psnr_value))

if len(psnr_values) == 0:
    sys.exit(1)

# Calculate threshold for bottom 50%
psnr_values.sort()
threshold_index = int(len(psnr_values) * 0.5)
threshold = psnr_values[threshold_index]

print(f'PSNR threshold (50th percentile): {threshold}', file=sys.stderr)

# Identify frames below threshold
retrain_frames = []
for frame_num, psnr_value in frame_data:
    if psnr_value < threshold:
        retrain_frames.append(frame_num)

# Write retrain frames to file
with open('$retrain_file', 'w') as f:
    for frame in retrain_frames:
        f.write(frame + '\n')

print(len(retrain_frames))
")
    
    echo "Identified $retrain_count frames for retraining"
    echo "Retrain list saved to: $retrain_file"
    
    # Cleanup
    rm -f "$psnr_file"
    
    if [ $retrain_count -eq 0 ]; then
        echo -e "${GREEN}All frames meet quality threshold - training complete${NC}"
        return 2
    fi
    
    return 0
}

# Function to create training script
create_training_script() {
    local iteration=$1
    local retrain_file="$OUTPUT_PATH/retrain_frames_iter${iteration}.txt"
    local training_script="$GAUSSIAN_DIR/hq_train_iter${iteration}.sh"
    
    echo "Creating training script: $training_script"
    
    # Copy quality-aware script and modify
    cp "$GAUSSIAN_DIR/train_gpu_quality.sh" "$training_script"
    
    # Modify paths
    sed -i "s|^BASE_DIR=.*|BASE_DIR=\"$DATASET_PATH/colmaps\"|" "$training_script"
    sed -i "s|^OUTPUT_BASE=.*|OUTPUT_BASE=\"$OUTPUT_PATH\"|" "$training_script"
    
    # Replace get_frames_for_gpu function
    sed -i '/^# Function to get frames for this GPU/,/^}/c\
# Function to get frames for this GPU from retrain list\
get_frames_for_gpu() {\
    local gpu_index=$1\
    local total_gpus=$2\
    local retrain_file="'"$retrain_file"'"\
    local frames=()\
    local frame_index=0\
    \
    if [ ! -f "$retrain_file" ]; then\
        echo "Error: Retrain file not found: $retrain_file" >&2\
        return 1\
    fi\
    \
    while read -r frame_num; do\
        if [ $((frame_index % total_gpus)) -eq $gpu_index ]; then\
            frames+=("$frame_num")\
        fi\
        ((frame_index++))\
    done < "$retrain_file"\
    \
    echo "${frames[@]}"\
}' "$training_script"
    
    chmod +x "$training_script"
    echo "Training script created: $training_script"
}

# Function to wait for training completion
wait_for_training() {
    local iteration=$1
    local session_prefix="hq_iter${iteration}"
    
    echo "Waiting for training iteration $iteration to complete..."
    while true; do
        active_sessions=0
        for gpu in "${GPU_ARRAY[@]}"; do
            if tmux has-session -t "${session_prefix}_gpu${gpu}" 2>/dev/null; then
                ((active_sessions++))
            fi
        done
        
        if [ $active_sessions -eq 0 ]; then
            echo -e "${GREEN}Training iteration $iteration completed!${NC}"
            break
        fi
        
        echo "Active sessions: $active_sessions/${#GPU_ARRAY[@]} - checking again in 60 seconds..."
        sleep 60
    done
}

# Function to run training iteration
run_training_iteration() {
    local iteration=$1
    local training_script="hq_train_iter${iteration}.sh"
    local session_prefix="hq_iter${iteration}"
    
    echo -e "${CYAN}=== Running Training Iteration $iteration ===${NC}"
    
    # Kill existing sessions
    for gpu in "${GPU_ARRAY[@]}"; do
        tmux kill-session -t "${session_prefix}_gpu${gpu}" 2>/dev/null || true
    done
    
    # Start training sessions
    echo "Starting tmux sessions for ${#GPU_ARRAY[@]} GPUs..."
    for i in "${!GPU_ARRAY[@]}"; do
        gpu=${GPU_ARRAY[$i]}
        session_name="${session_prefix}_gpu${gpu}"
        echo "Starting session: $session_name"
        
        tmux new-session -d -s "$session_name" -c "$GAUSSIAN_DIR" \
            "bash -c 'echo \"Starting GPU $gpu training iteration $iteration...\"; ./$training_script $gpu $i ${#GPU_ARRAY[@]}; echo \"GPU $gpu iteration $iteration completed.\"; sleep 5'"
        
        echo -e "${GREEN}✓${NC} GPU $gpu session started: $session_name"
    done
    
    echo -e "${GREEN}All GPU sessions started successfully!${NC}"
    
    # Wait for completion
    wait_for_training "$iteration"
    
    # Cleanup
    rm -f "$GAUSSIAN_DIR/$training_script"
}

# Main loop
echo -e "${GREEN}Starting high-quality iterative training process...${NC}"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_PATH"
echo "GPUs: ${GPU_ARRAY[*]}"
echo "Max iterations: $MAX_ITERATIONS"
echo ""

for ((iteration=1; iteration<=MAX_ITERATIONS; iteration++)); do
    echo ""
    echo -e "${YELLOW}=== Iteration $iteration/$MAX_ITERATIONS ===${NC}"
    
    # Analyze PSNR results
    if ! analyze_psnr_results "$iteration"; then
        case $? in
            1)
                echo -e "${RED}PSNR analysis failed - stopping training${NC}"
                break
                ;;
            2)
                echo -e "${GREEN}All frames meet quality threshold - training complete${NC}"
                break
                ;;
        esac
    fi
    
    # Create and run training
    create_training_script "$iteration"
    run_training_iteration "$iteration"
    
    echo -e "${GREEN}Iteration $iteration completed${NC}"
done

echo ""
echo -e "${BLUE}=== Training Summary ===${NC}"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_PATH"
echo "GPUs used: ${GPU_ARRAY[*]}"
echo "Iterations completed: $iteration"
echo -e "${GREEN}High-quality iterative training completed successfully!${NC}"

# Keep session alive
echo ""
echo "Training completed. Press any key to exit."
read
