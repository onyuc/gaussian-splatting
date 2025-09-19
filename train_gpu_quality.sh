#!/bin/bash

# Quality-aware GPU training script with PSNR comparison
# Usage: ./train_gpu_quality.sh <gpu_id> [gpu_index] [total_gpus]

GPU_ID=$1
GPU_INDEX=${2:-$1}  # Default to GPU_ID if not provided (backward compatibility)
TOTAL_GPUS=${3:-4}  # Default to 4 GPUs if not provided (backward compatibility)

# These will be replaced by hq_training.sh
BASE_DIR="/mnt/HDD/onyuc/dataset/N3DV/cut_roasted_beef/colmaps"
OUTPUT_BASE="/mnt/HDD/onyuc/gaussian_splatting/output/cut_roasted_beef"

GAUSSIAN_DIR="/home/onyuc/Workspace/gaussian-splatting"
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

# Function to get frames for this GPU from retrain list (will be replaced)
get_frames_for_gpu() {
    local gpu_index=$1
    local total_gpus=$2
    local retrain_file="/mnt/HDD/onyuc/gaussian_splatting/output/cut_roasted_beef/retrain_frames_iter1.txt"
    local frames=()
    local frame_index=0
    
    if [ ! -f "$retrain_file" ]; then
        echo "Error: Retrain file not found: $retrain_file" >&2
        return 1
    fi
    
    while read -r frame_num; do
        if [ $((frame_index % total_gpus)) -eq $gpu_index ]; then
            frames+=("$frame_num")
        fi
        ((frame_index++))
    done < "$retrain_file"
    
    echo "${frames[@]}"
}

# Function to get PSNR from results.json
get_psnr_from_results() {
    local results_file=$1
    
    if [ ! -f "$results_file" ]; then
        echo "0"  # Return 0 if no results file
        return
    fi
    
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
    else:
        print('0')
except:
    print('0')
" 2>/dev/null)
    
    echo "${psnr_value:-0}"
}

# Function to compare PSNR values
compare_psnr() {
    local psnr1=$1
    local psnr2=$2
    
    # Use python for reliable floating point comparison
    local result=$(python3 -c "print(1 if float('$psnr1') > float('$psnr2') else 0)" 2>/dev/null)
    echo "${result:-0}"
}

# Function to train a single frame with quality comparison
train_frame() {
    local frame_num=$1
    local frame_name="frame_$frame_num"
    local source_path="$BASE_DIR/$frame_name"
    local model_path="$OUTPUT_BASE/F$frame_num"
    local temp_path="$OUTPUT_BASE/F${frame_num}_temp"
    
    log "Starting quality-aware training for $frame_name"
    
    # Check if source exists
    if [ ! -d "$source_path" ]; then
        log "ERROR: Source path does not exist: $source_path"
        return 1
    fi
    
    # Get existing PSNR if available
    local existing_psnr=0
    if [ -f "$model_path/results.json" ]; then
        existing_psnr=$(get_psnr_from_results "$model_path/results.json")
        log "Existing PSNR for $frame_name: $existing_psnr"
    else
        log "No existing results for $frame_name - will train"
    fi
    
    # Create temporary output directory
    mkdir -p "$temp_path"
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    # Training command to temporary location with floater reduction parameters
    cd "$GAUSSIAN_DIR"
    log "Training $frame_name to temporary location: $temp_path (with floater reduction)"
    timeout 7200 python train.py \
        -s "$source_path" \
        -m "$temp_path" \
        --opacity_reset_interval $OPACITY_RESET \
        --resolution $RESOLUTION \
        --eval \
        --disable_viewer \
        > /dev/null 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        if [ $exit_code -eq 124 ]; then
            log "Training timeout for $frame_name"
        else
            log "Training failed for $frame_name (exit code: $exit_code)"
        fi
        # Cleanup temp directory
        rm -rf "$temp_path"
        return 1
    fi
    
    log "Training completed for $frame_name, now comparing quality..."
    return 0
}

# Function to render a frame (works with temp path)
render_frame() {
    local frame_num=$1
    local use_temp=${2:-false}
    local frame_name="frame_$frame_num"
    
    if [ "$use_temp" = true ]; then
        local model_path="$OUTPUT_BASE/F${frame_num}_temp"
        log "Starting rendering for $frame_name (temp)"
    else
        local model_path="$OUTPUT_BASE/F$frame_num"
        log "Starting rendering for $frame_name"
    fi
    
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
        > /dev/null 2>&1
    
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

# Function to evaluate metrics (works with temp path)
evaluate_metrics() {
    local frame_num=$1
    local use_temp=${2:-false}
    local frame_name="frame_$frame_num"
    
    if [ "$use_temp" = true ]; then
        local model_path="$OUTPUT_BASE/F${frame_num}_temp"
        log "Starting metrics evaluation for $frame_name (temp)"
    else
        local model_path="$OUTPUT_BASE/F$frame_num"
        log "Starting metrics evaluation for $frame_name"
    fi
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    # Metrics command
    cd "$GAUSSIAN_DIR"
    timeout 600 python metrics.py \
        -m "$model_path" \
        > /dev/null 2>&1
    
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

# Function to process a complete frame with quality comparison
process_frame() {
    local frame_num=$1
    local frame_name="frame_$frame_num"
    local model_path="$OUTPUT_BASE/F$frame_num"
    local temp_path="$OUTPUT_BASE/F${frame_num}_temp"
    
    log "Starting complete processing for $frame_name"
    local start_time=$(date +%s)
    
    # Step 1: Training to temp location
    if ! train_frame "$frame_num"; then
        log "Failed to train $frame_name"
        echo "$frame_num" >> "$LOG_DIR/gpu_${GPU_ID}_failed.txt"
        return 1
    fi
    
    # Step 2: Rendering temp results
    if ! render_frame "$frame_num" true; then
        log "Failed to render $frame_name (temp), cleaning up"
        rm -rf "$temp_path"
        echo "$frame_num" >> "$LOG_DIR/gpu_${GPU_ID}_failed.txt"
        return 1
    fi
    
    # Step 3: Evaluate metrics for temp results
    if ! evaluate_metrics "$frame_num" true; then
        log "Failed to evaluate metrics for $frame_name (temp), cleaning up"
        rm -rf "$temp_path"
        echo "$frame_num" >> "$LOG_DIR/gpu_${GPU_ID}_failed.txt"
        return 1
    fi
    
    # Step 4: Compare PSNR values
    local existing_psnr=0
    if [ -f "$model_path/results.json" ]; then
        existing_psnr=$(get_psnr_from_results "$model_path/results.json")
    fi
    
    local new_psnr=$(get_psnr_from_results "$temp_path/results.json")
    
    log "PSNR comparison for $frame_name: existing=$existing_psnr, new=$new_psnr"
    
    # Compare PSNR values
    local is_better=$(compare_psnr "$new_psnr" "$existing_psnr")
    
    if [ "$is_better" = "1" ]; then
        log "New result is better! Replacing existing model for $frame_name"
        
        # Backup existing if it exists
        if [ -d "$model_path" ]; then
            rm -rf "${model_path}_backup" 2>/dev/null || true
            mv "$model_path" "${model_path}_backup"
        fi
        
        # Move temp to final location
        mv "$temp_path" "$model_path"
        
        log "Successfully updated $frame_name (PSNR: $existing_psnr → $new_psnr)"
        echo "$frame_num" >> "$LOG_DIR/gpu_${GPU_ID}_improved.txt"
    else
        log "Existing result is better, keeping original for $frame_name (PSNR: $existing_psnr vs $new_psnr)"
        
        # Remove temp directory
        rm -rf "$temp_path"
        
        echo "$frame_num" >> "$LOG_DIR/gpu_${GPU_ID}_kept_original.txt"
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
    
    log "Starting quality-aware GPU $GPU_ID processing (index $GPU_INDEX/$TOTAL_GPUS)"
    log "Log file: $LOG_FILE"
    
    # Get frames assigned to this GPU
    frames=($(get_frames_for_gpu $GPU_INDEX $TOTAL_GPUS))
    total_frames=${#frames[@]}
    
    log "Assigned $total_frames frames: ${frames[@]:0:5}..."
    
    # Initialize counters
    completed=0
    failed=0
    improved=0
    kept_original=0
    
    # Clear previous results
    rm -f "$LOG_DIR/gpu_${GPU_ID}_completed.txt"
    rm -f "$LOG_DIR/gpu_${GPU_ID}_failed.txt"
    rm -f "$LOG_DIR/gpu_${GPU_ID}_improved.txt"
    rm -f "$LOG_DIR/gpu_${GPU_ID}_kept_original.txt"
    
    # Process each frame
    for i in "${!frames[@]}"; do
        frame_num=${frames[$i]}
        current=$((i + 1))
        
        log "Processing frame $current/$total_frames: frame_$frame_num"
        
        if process_frame "$frame_num"; then
            ((completed++))
            
            # Count improvements vs kept originals
            if grep -q "^$frame_num$" "$LOG_DIR/gpu_${GPU_ID}_improved.txt" 2>/dev/null; then
                ((improved++))
            elif grep -q "^$frame_num$" "$LOG_DIR/gpu_${GPU_ID}_kept_original.txt" 2>/dev/null; then
                ((kept_original++))
            fi
        else
            ((failed++))
        fi
        
        # Progress report every 5 frames
        if [ $((current % 5)) -eq 0 ]; then
            log "Progress: $current/$total_frames frames processed (Completed: $completed, Failed: $failed, Improved: $improved, Kept: $kept_original)"
        fi
    done
    
    log "Quality-aware GPU $GPU_ID processing completed!"
    log "Total: $total_frames, Completed: $completed, Failed: $failed"
    log "Quality results: Improved: $improved, Kept original: $kept_original"
    
    # Save final summary
    echo "GPU $GPU_ID Quality Summary:" > "$LOG_DIR/gpu_${GPU_ID}_summary.txt"
    echo "Total frames: $total_frames" >> "$LOG_DIR/gpu_${GPU_ID}_summary.txt"
    echo "Completed: $completed" >> "$LOG_DIR/gpu_${GPU_ID}_summary.txt"
    echo "Failed: $failed" >> "$LOG_DIR/gpu_${GPU_ID}_summary.txt"
    echo "Improved: $improved" >> "$LOG_DIR/gpu_${GPU_ID}_summary.txt"
    echo "Kept original: $kept_original" >> "$LOG_DIR/gpu_${GPU_ID}_summary.txt"
    echo "Completion time: $(date)" >> "$LOG_DIR/gpu_${GPU_ID}_summary.txt"
}

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_splatting

# Run main function
main "$@"
