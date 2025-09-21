#!/bin/bash

# High-Quality Iterative Training Script
# Runs independently in tmux with automatic PSNR analysis and retraining

GAUSSIAN_DIR="/home/onyuk/Workspace/gaussian-splatting"
SESSION_NAME="hq_training"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Global variables
declare -a GPU_ARRAY
DATASET_PATH=""
OUTPUT_PATH=""
LOG_DIR=""
MAX_ITERATIONS=10
PSNR_THRESHOLD_PERCENTILE=50

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}High-Quality Iterative Training System${NC}"
echo -e "${BLUE}================================================${NC}"

# Function to get dataset path from user
get_dataset_path() {
    echo -e "${CYAN}=== Dataset Configuration ===${NC}"
    echo ""
    
    while true; do
        echo -e "${YELLOW}Enter the dataset path (should contain 'colmaps' directory):${NC}"
        echo "Example: /data1/onyuk/dataset/N3DV/cut_roasted_beef"
        read -p "Dataset path: " dataset_input
        
        # Remove trailing slash
        dataset_input="${dataset_input%/}"
        
        # Check if path exists
        if [ ! -d "$dataset_input" ]; then
            echo -e "${RED}Error: Directory does not exist: $dataset_input${NC}"
            continue
        fi
        
        # Check if colmaps directory exists
        if [ ! -d "$dataset_input/colmaps" ]; then
            echo -e "${RED}Error: 'colmaps' directory not found in: $dataset_input${NC}"
            echo "Expected structure: $dataset_input/colmaps/frame_XXXX/"
            continue
        fi
        
        # Count available frames
        frame_count=$(find "$dataset_input/colmaps" -maxdepth 1 -name "frame_*" -type d | wc -l)
        if [ $frame_count -eq 0 ]; then
            echo -e "${RED}Error: No frame directories found in: $dataset_input/colmaps${NC}"
            continue
        fi
        
        DATASET_PATH="$dataset_input"
        echo -e "${GREEN}✓ Dataset path validated: $DATASET_PATH${NC}"
        echo -e "${GREEN}✓ Found $frame_count frame directories${NC}"
        break
    done
}

# Function to configure output path
get_output_path() {
    echo -e "${CYAN}=== Output Configuration ===${NC}"
    echo ""
    
    # Extract dataset name from path
    dataset_name=$(basename "$DATASET_PATH")
    
    # Suggest default output path
    default_output="/data1/onyuk/gaussian_splatting/output/$dataset_name"
    
    echo -e "${YELLOW}Suggested output path:${NC} $default_output"
    echo ""
    
    while true; do
        echo -e "${YELLOW}Enter output path (or press Enter for default):${NC}"
        read -p "Output path: " output_input
        
        # Use default if empty
        if [ -z "$output_input" ]; then
            output_input="$default_output"
        fi
        
        # Remove trailing slash
        output_input="${output_input%/}"
        
        # Check if parent directory exists and is writable
        parent_dir=$(dirname "$output_input")
        if [ ! -d "$parent_dir" ]; then
            echo -e "${RED}Error: Parent directory does not exist: $parent_dir${NC}"
            continue
        fi
        
        if [ ! -w "$parent_dir" ]; then
            echo -e "${RED}Error: No write permission to: $parent_dir${NC}"
            continue
        fi
        
        OUTPUT_PATH="$output_input"
        echo -e "${GREEN}✓ Output path configured: $OUTPUT_PATH${NC}"
        
        # Create output directory if it doesn't exist
        mkdir -p "$OUTPUT_PATH"
        
        # Create log directory structure
        LOG_DIR="$OUTPUT_PATH/logs"
        mkdir -p "$LOG_DIR/iterations"
        mkdir -p "$LOG_DIR/gpu_sessions"
        mkdir -p "$LOG_DIR/psnr_analysis"
        echo -e "${GREEN}✓ Log directories created: $LOG_DIR${NC}"
        
        break
    done
}

# Function to get GPU input from user
get_gpu_input() {
    echo -e "${CYAN}=== GPU Configuration ===${NC}"
    echo ""
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
    echo ""
}

# Function to get PSNR threshold
get_psnr_threshold() {
    echo -e "${CYAN}=== PSNR Threshold Configuration ===${NC}"
    echo ""
    
    while true; do
        echo -e "${YELLOW}Enter PSNR threshold percentile (1-99, default: 50):${NC}"
        echo "Lower values = more frames retrained (stricter quality)"
        echo "Higher values = fewer frames retrained (more lenient)"
        read -p "Threshold percentile: " threshold_input
        
        # Use default if empty
        if [ -z "$threshold_input" ]; then
            threshold_input=50
        fi
        
        # Validate input
        if [[ "$threshold_input" =~ ^[1-9]$|^[1-9][0-9]$ ]] && [ "$threshold_input" -le 99 ]; then
            PSNR_THRESHOLD_PERCENTILE=$threshold_input
            break
        else
            echo -e "${RED}Error: Please enter a number between 1 and 99${NC}"
        fi
    done
    
    echo -e "${GREEN}✓ PSNR threshold percentile: $PSNR_THRESHOLD_PERCENTILE%${NC}"
    echo ""
}

# Function to get max iterations
get_max_iterations() {
    echo -e "${CYAN}=== Training Parameters ===${NC}"
    echo ""
    
    while true; do
        echo -e "${YELLOW}Enter maximum number of iterations (1-50, default: 10):${NC}"
        read -p "Max iterations: " max_input
        
        # Use default if empty
        if [ -z "$max_input" ]; then
            max_input=10
        fi
        
        # Validate input
        if [[ "$max_input" =~ ^[1-9]$|^[1-4][0-9]$|^50$ ]]; then
            MAX_ITERATIONS=$max_input
            break
        else
            echo -e "${RED}Error: Please enter a number between 1 and 50${NC}"
        fi
    done
    
    echo -e "${GREEN}✓ Maximum iterations: $MAX_ITERATIONS${NC}"
    echo ""
}

# Function to analyze PSNR results and identify poor performers
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
    echo "Collected frames: $collected_frames/$total_frames" >> "$LOG_DIR/iterations/psnr_analysis.log"
    
    if [ $collected_frames -lt 10 ]; then
        echo -e "${RED}Error: Not enough PSNR data collected ($collected_frames frames)${NC}"
        echo "Need at least 10 frames with completed metrics"
        echo "ERROR: Insufficient data - $collected_frames frames" >> "$LOG_DIR/iterations/psnr_analysis.log"
        return 1
    fi
    
    # Identify frames below threshold using Python (no bc needed)
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

# Calculate threshold for specified percentile
psnr_values.sort()
threshold_index = int(len(psnr_values) * $PSNR_THRESHOLD_PERCENTILE / 100.0)
threshold = psnr_values[threshold_index]

print(f'PSNR threshold ({$PSNR_THRESHOLD_PERCENTILE}th percentile): {threshold}', file=sys.stderr)

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
    
    # Log results
    echo "Retrain count: $retrain_count" >> "$LOG_DIR/iterations/psnr_analysis.log"
    echo "Threshold used: $PSNR_THRESHOLD_PERCENTILE%" >> "$LOG_DIR/iterations/psnr_analysis.log"
    echo "---" >> "$LOG_DIR/iterations/psnr_analysis.log"
    
    # Cleanup
    rm -f "$psnr_file"
    
    if [ $retrain_count -eq 0 ]; then
        echo -e "${GREEN}All frames meet quality threshold - training complete${NC}"
        return 2  # Special return code for "no retraining needed"
    fi
    
    return 0
}

# Function to create training script for specific frames
create_training_script() {
    local iteration=$1
    local retrain_file="$OUTPUT_PATH/retrain_frames_iter${iteration}.txt"
    local training_script="$GAUSSIAN_DIR/hq_train_iter${iteration}.sh"
    
    echo "Creating training script: $training_script"
    
    # Copy original script and modify for specific frames
    cp "$GAUSSIAN_DIR/train_gpu.sh" "$training_script"
    
    # Modify the script to use custom dataset and output paths
    sed -i "s|^BASE_DIR=.*|BASE_DIR=\"$DATASET_PATH/colmaps\"|" "$training_script"
    sed -i "s|^OUTPUT_BASE=.*|OUTPUT_BASE=\"$OUTPUT_PATH\"|" "$training_script"
    
    # Replace the get_frames_for_gpu function to use retrain list
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

# Function to run training iteration
run_training_iteration() {
    local iteration=$1
    local training_script="hq_train_iter${iteration}.sh"
    local session_prefix="hq_iter${iteration}"
    
    echo -e "${CYAN}=== Running Training Iteration $iteration ===${NC}"
    
    # Kill existing sessions if they exist
    echo "Cleaning up existing tmux sessions..."
    for gpu in "${GPU_ARRAY[@]}"; do
        tmux kill-session -t "${session_prefix}_gpu${gpu}" 2>/dev/null || true
    done
    
    # Start tmux sessions for each selected GPU
    echo "Starting tmux sessions for ${#GPU_ARRAY[@]} GPUs..."
    for i in "${!GPU_ARRAY[@]}"; do
        gpu=${GPU_ARRAY[$i]}
        session_name="${session_prefix}_gpu${gpu}"
        echo "Starting session: $session_name"
        
        # Create new tmux session and run the training script with logging
        tmux new-session -d -s "$session_name" -c "$GAUSSIAN_DIR" \
            "bash -c 'exec > >(tee -a \"$LOG_DIR/gpu_sessions/gpu${gpu}_iter${iteration}.log\") 2>&1; echo \"Starting GPU $gpu training iteration $iteration...\"; ./$training_script $gpu $i ${#GPU_ARRAY[@]}; echo \"GPU $gpu iteration $iteration completed.\"; sleep 5'"
        
        echo -e "${GREEN}✓${NC} GPU $gpu session started: $session_name"
    done
    
    echo ""
    echo -e "${GREEN}All GPU sessions started successfully!${NC}"
    
    # Wait for completion
    echo "Waiting for training iteration $iteration to complete..."
    wait_for_completion "$session_prefix"
    
    # Cleanup training script
    rm -f "$GAUSSIAN_DIR/$training_script"
}

# Function to wait for all training sessions to complete
wait_for_completion() {
    local session_prefix=$1
    
    echo "Monitoring training progress..."
    while true; do
        active_sessions=0
        for gpu in "${GPU_ARRAY[@]}"; do
            if tmux has-session -t "${session_prefix}_gpu${gpu}" 2>/dev/null; then
                ((active_sessions++))
            fi
        done
        
        if [ $active_sessions -eq 0 ]; then
            echo -e "${GREEN}All training sessions completed!${NC}"
            break
        fi
        
        echo "Active sessions: $active_sessions/${#GPU_ARRAY[@]} - checking again in 60 seconds..."
        sleep 60
    done
}

# Create master control script that runs in tmux
create_master_script() {
    local master_script="$GAUSSIAN_DIR/hq_master.sh"
    
    cat > "$master_script" << 'EOF'
#!/bin/bash

# Master HQ training script that runs in tmux
DATASET_PATH="__DATASET_PATH__"
OUTPUT_PATH="__OUTPUT_PATH__"
MAX_ITERATIONS=__MAX_ITERATIONS__
GPU_ARRAY=(__GPU_ARRAY__)

GAUSSIAN_DIR="/home/onyuk/Workspace/gaussian-splatting"
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

# Initialize progress log
LOG_DIR="__LOG_DIR__"
echo "=== HQ Training Progress Log ===" > "$LOG_DIR/progress.log"
echo "Start time: $(date)" >> "$LOG_DIR/progress.log"
echo "Dataset: $DATASET_PATH" >> "$LOG_DIR/progress.log"
echo "Output: $OUTPUT_PATH" >> "$LOG_DIR/progress.log"
echo "GPUs: ${GPU_ARRAY[*]}" >> "$LOG_DIR/progress.log"
echo "Max iterations: $MAX_ITERATIONS" >> "$LOG_DIR/progress.log"
echo "PSNR threshold: __PSNR_THRESHOLD_PERCENTILE__%%" >> "$LOG_DIR/progress.log"
echo "---" >> "$LOG_DIR/progress.log"

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

# Calculate threshold for specified percentile
psnr_values.sort()
threshold_index = int(len(psnr_values) * __PSNR_THRESHOLD_PERCENTILE__ / 100.0)
threshold = psnr_values[threshold_index]

print(f'PSNR threshold (__PSNR_THRESHOLD_PERCENTILE__th percentile): {threshold}', file=sys.stderr)

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
            "bash -c 'exec > >(tee -a \"__LOG_DIR__/gpu_sessions/gpu${gpu}_iter${iteration}.log\") 2>&1; echo \"Starting GPU $gpu training iteration $iteration...\"; ./$training_script $gpu $i ${#GPU_ARRAY[@]}; echo \"GPU $gpu iteration $iteration completed.\"; sleep 5'"
        
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
    
    # Log iteration start
    echo "Iteration $iteration started at $(date)" >> "$LOG_DIR/progress.log"
    
    # Analyze PSNR results
    if ! analyze_psnr_results "$iteration"; then
        case $? in
            1)
                echo -e "${RED}PSNR analysis failed - stopping training${NC}"
                echo "Iteration $iteration FAILED - PSNR analysis error at $(date)" >> "$LOG_DIR/progress.log"
                break
                ;;
            2)
                echo -e "${GREEN}All frames meet quality threshold - training complete${NC}"
                echo "Training COMPLETED - All frames meet threshold at $(date)" >> "$LOG_DIR/progress.log"
                break
                ;;
        esac
    fi
    
    # Create and run training
    create_training_script "$iteration"
    run_training_iteration "$iteration"
    
    echo -e "${GREEN}Iteration $iteration completed${NC}"
    echo "Iteration $iteration completed at $(date)" >> "$LOG_DIR/progress.log"
done

echo ""
echo -e "${BLUE}=== Training Summary ===${NC}"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_PATH"
echo "GPUs used: ${GPU_ARRAY[*]}"
echo "Iterations completed: $iteration"
echo -e "${GREEN}High-quality iterative training completed successfully!${NC}"

# Final progress log
echo "=== TRAINING COMPLETED ===" >> "$LOG_DIR/progress.log"
echo "End time: $(date)" >> "$LOG_DIR/progress.log"
echo "Total iterations completed: $iteration" >> "$LOG_DIR/progress.log"
echo "GPUs used: ${GPU_ARRAY[*]}" >> "$LOG_DIR/progress.log"

# Keep session alive
echo ""
echo "Training completed. Press any key to exit."
read
EOF

    # Replace placeholders
    sed -i "s|__DATASET_PATH__|$DATASET_PATH|g" "$master_script"
    sed -i "s|__OUTPUT_PATH__|$OUTPUT_PATH|g" "$master_script"
    sed -i "s|__LOG_DIR__|$LOG_DIR|g" "$master_script"
    sed -i "s|__MAX_ITERATIONS__|$MAX_ITERATIONS|g" "$master_script"
    sed -i "s|__PSNR_THRESHOLD_PERCENTILE__|$PSNR_THRESHOLD_PERCENTILE|g" "$master_script"
    sed -i "s|__GPU_ARRAY__|${GPU_ARRAY[*]}|g" "$master_script"
    
    chmod +x "$master_script"
    echo "Master script created: $master_script"
}

# Main execution function
main() {
    # Change to gaussian-splatting directory
    cd "$GAUSSIAN_DIR"
    
    # Check prerequisites
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    if ! command -v tmux &> /dev/null; then
        echo -e "${RED}Error: tmux is not installed${NC}"
        exit 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: nvidia-smi not available${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ All prerequisites met${NC}"
    echo ""
    
    # Get configuration
    get_dataset_path
    get_output_path
    get_psnr_threshold
    get_max_iterations
    get_gpu_input
    
    # Show final configuration
    echo -e "${BLUE}=== Final Configuration ===${NC}"
    echo "Dataset path: $DATASET_PATH"
    echo "Output path: $OUTPUT_PATH"
    echo "Log directory: $LOG_DIR"
    echo "GPUs: ${GPU_ARRAY[*]}"
    echo "Max iterations: $MAX_ITERATIONS"
    echo "PSNR threshold: $PSNR_THRESHOLD_PERCENTILE%"
    echo ""
    
    read -p "Proceed with high-quality iterative training? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Training cancelled by user"
        exit 0
    fi
    
    # Check existing results
    existing_results=0
    for frame_num in $(seq -f "%04g" 0 299); do
        if [ -f "$OUTPUT_PATH/F$frame_num/results.json" ]; then
            ((existing_results++))
        fi
    done
    
    echo "Found $existing_results/300 existing training results"
    
    if [ $existing_results -lt 100 ]; then
        echo -e "${RED}Error: Not enough existing results ($existing_results/300)${NC}"
        echo "Please run initial training first"
        exit 1
    fi
    
    # Create master script and run in tmux
    create_master_script
    
    echo ""
    echo -e "${GREEN}Starting HQ training in tmux session...${NC}"
    echo "Session name: hq_master"
    echo ""
    echo "To monitor progress:"
    echo "  tmux attach -t hq_master"
    echo "  Ctrl+B, D to detach"
    echo ""
    echo "Log files will be saved to:"
    echo "  Master log: $LOG_DIR/hq_master.log"
    echo "  GPU sessions: $LOG_DIR/gpu_sessions/"
    echo "  PSNR analysis: $LOG_DIR/iterations/psnr_analysis.log"
    echo ""
    
    # Start master session with logging
    tmux new-session -d -s hq_master -c "$GAUSSIAN_DIR" \
        "bash -c 'exec > >(tee -a \"$LOG_DIR/hq_master.log\") 2>&1; ./hq_master.sh'"
    
    echo -e "${GREEN}HQ training started in background!${NC}"
    echo "Use 'tmux attach -t hq_master' to monitor progress"
}

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_splatting

# Run main function
main "$@"
