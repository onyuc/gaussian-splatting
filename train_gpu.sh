#!/bin/bash

# Individual GPU training script
# Usage: ./train_gpu.sh <gpu_id> [gpu_index] [total_gpus]

GPU_ID=$1
GPU_INDEX=${2:-$1}  # Default to GPU_ID if not provided (backward compatibility)
TOTAL_GPUS=${3:-4}  # Default to 4 GPUs if not provided (backward compatibility)
# select Dataset: coffee_martini
# BASE_DIR="/home/onyuk/Dataset/N3DV/coffee_martini/colmaps"
# OUTPUT_BASE="/home/onyuk/Workspace/gaussian-splatting/output/coffee_martini"
BASE_DIR="/data1/onyuk/Dataset/N3DV/cook_spinach/colmaps"
OUTPUT_BASE="/data1/onyuk/gaussian_splatting/output/cook_spinach"

GAUSSIAN_DIR="/home/onyuk/Workspace/gaussian-splatting"
LOG_DIR="$GAUSSIAN_DIR/logs"

# Training parameters
OPACITY_RESET=2000000
RESOLUTION=1352

# Create log directory
mkdir -p "$LOG_DIR"

# Log file for this GPU
LOG_FILE="$LOG_DIR/gpu_${GPU_ID}.log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU$GPU_ID: $1" | tee -a "$LOG_FILE"
}

# Function to get frames for this GPU (modulo distribution)
get_frames_for_gpu() {
    local gpu_index=$1
    local total_gpus=$2
    local frames=()
    for i in $(seq 00 299); do
        if [ $((i % total_gpus)) -eq $gpu_index ]; then
            frames+=($(printf "%04d" $i))
        fi
    done
    echo "${frames[@]}"
}

# Function to train a single frame
train_frame() {
    local frame_num=$1
    local frame_name="frame_$frame_num"
    local source_path="$BASE_DIR/$frame_name"
    local model_path="$OUTPUT_BASE/F$frame_num"
    
    log "Starting training for $frame_name"
    
    # Check if source exists
    if [ ! -d "$source_path" ]; then
        log "ERROR: Source path does not exist: $source_path"
        return 1
    fi
    
    # Create output directory
    mkdir -p "$model_path"
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    # Training command
    cd "$GAUSSIAN_DIR"
    timeout 7200 python train.py \
        -s "$source_path" \
        -m "$model_path" \
        --opacity_reset_interval $OPACITY_RESET \
        --resolution $RESOLUTION \
        --eval \
        --disable_viewer
        # >> "$LOG_FILE" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log "Training completed successfully for $frame_name"
        return 0
    elif [ $exit_code -eq 124 ]; then
        log "Training timeout for $frame_name"
        return 1
    else
        log "Training failed for $frame_name (exit code: $exit_code)"
        return 1
    fi
}

# Function to render a frame
render_frame() {
    local frame_num=$1
    local frame_name="frame_$frame_num"
    local model_path="$OUTPUT_BASE/F$frame_num"
    
    log "Starting rendering for $frame_name"
    
    # Check if model exists
    if [ ! -d "$model_path" ]; then
        log "ERROR: Model path does not exist: $model_path"
        return 1
    fi
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    # Rendering command
    cd "$GAUSSIAN_DIR"
    timeout 1800 python render.py \
        -m "$model_path" \
        >> "$LOG_FILE" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log "Rendering completed successfully for $frame_name"
        return 0
    elif [ $exit_code -eq 124 ]; then
        log "Rendering timeout for $frame_name"
        return 1
    else
        log "Rendering failed for $frame_name (exit code: $exit_code)"
        return 1
    fi
}

# Function to evaluate metrics
evaluate_metrics() {
    local frame_num=$1
    local frame_name="frame_$frame_num"
    local model_path="$OUTPUT_BASE/F$frame_num"
    
    log "Starting metrics evaluation for $frame_name"
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    # Metrics command
    cd "$GAUSSIAN_DIR"
    timeout 600 python metrics.py \
        -m "$model_path" \
        >> "$LOG_FILE" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log "Metrics evaluation completed successfully for $frame_name"
        return 0
    elif [ $exit_code -eq 124 ]; then
        log "Metrics evaluation timeout for $frame_name"
        return 1
    else
        log "Metrics evaluation failed for $frame_name (exit code: $exit_code)"
        return 1
    fi
}

# Function to process a complete frame (train -> render -> metrics)
process_frame() {
    local frame_num=$1
    local frame_name="frame_$frame_num"
    
    log "Starting complete processing for $frame_name"
    local start_time=$(date +%s)
    
    # Step 1: Training
    if ! train_frame "$frame_num"; then
        log "Failed to train $frame_name, skipping render and metrics"
        echo "$frame_num" >> "$LOG_DIR/gpu_${GPU_ID}_failed.txt"
        return 1
    fi
    
    # Step 2: Rendering
    if ! render_frame "$frame_num"; then
        log "Failed to render $frame_name, skipping metrics"
        echo "$frame_num" >> "$LOG_DIR/gpu_${GPU_ID}_failed.txt"
        return 1
    fi
    
    # Step 3: Metrics
    if ! evaluate_metrics "$frame_num"; then
        log "Failed to evaluate metrics for $frame_name"
        echo "$frame_num" >> "$LOG_DIR/gpu_${GPU_ID}_failed.txt"
        return 1
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "Complete processing finished for $frame_name in ${duration}s"
    echo "$frame_num" >> "$LOG_DIR/gpu_${GPU_ID}_completed.txt"
    return 0
}

# Main execution
main() {
    if [ -z "$GPU_ID" ]; then
        echo "Usage: $0 <gpu_id>"
        exit 1
    fi
    
    log "Starting GPU $GPU_ID processing (index $GPU_INDEX/$TOTAL_GPUS)"
    log "Log file: $LOG_FILE"
    
    # Get frames assigned to this GPU
    frames=($(get_frames_for_gpu $GPU_INDEX $TOTAL_GPUS))
    total_frames=${#frames[@]}
    
    log "Assigned $total_frames frames: ${frames[@]:0:5}..."
    
    # Initialize counters
    completed=0
    failed=0
    
    # Clear previous results
    rm -f "$LOG_DIR/gpu_${GPU_ID}_completed.txt"
    rm -f "$LOG_DIR/gpu_${GPU_ID}_failed.txt"
    
    # Process each frame
    for i in "${!frames[@]}"; do
        frame_num=${frames[$i]}
        current=$((i + 1))
        
        log "Processing frame $current/$total_frames: frame_$frame_num"
        
        if process_frame "$frame_num"; then
            ((completed++))
        else
            ((failed++))
        fi
        
        # Progress report every 5 frames
        if [ $((current % 5)) -eq 0 ]; then
            log "Progress: $current/$total_frames frames processed (Completed: $completed, Failed: $failed)"
        fi
    done
    
    log "GPU $GPU_ID processing completed!"
    log "Total: $total_frames, Completed: $completed, Failed: $failed"
    
    # Save final summary
    echo "GPU $GPU_ID Summary:" > "$LOG_DIR/gpu_${GPU_ID}_summary.txt"
    echo "Total frames: $total_frames" >> "$LOG_DIR/gpu_${GPU_ID}_summary.txt"
    echo "Completed: $completed" >> "$LOG_DIR/gpu_${GPU_ID}_summary.txt"
    echo "Failed: $failed" >> "$LOG_DIR/gpu_${GPU_ID}_summary.txt"
    echo "Completion time: $(date)" >> "$LOG_DIR/gpu_${GPU_ID}_summary.txt"
}

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_splatting

# Run main function
main "$@"
