#!/usr/bin/env bash
set -u  # 미정의 변수 사용 방지

#IGS/RaDe 이동
gs_path="/scratch/rchkl2380/Workspace/gaussian-splatting/freetimegs_finetune"
current_path=$(pwd)

cd $gs_path

base_path="/scratch2/y5ryan/Common/4D_SOTA/dataset/N3DV"
dataset_name="cook_spinach"
freetimegs_path="/scratch2/y5ryan/Common/4D_SOTA/pretrained/FTGS/${dataset_name}/gaussian_dict_${dataset_name}.pt"
output_path="/scratch2/y5ryan/Common/4D_SOTA/output/N3DV"
extra_name=""

# tmux 활성화후 srun으로 gpu 배정받고 실행
# srun --gres=gpu:8 --time=12:00:00 --pty bash -i
gpu_ea=8          # 사용할 GPU 개수 (GPU ID가 0..gpu_ea-1이라고 가정)
start_num=0
end_num=299
ITERS=(1 6000)
# ITERS=(1 10 50 100 200 300 400 500 600 700 800 900 1000 1500)


run_one() {
  local frame_num="$1"
  local gpu_id="$2"

  local source_path="${base_path}/${dataset_name}/colmap_${frame_num}"
  local model_path="${output_path}/${dataset_name}/frame_${frame_num}${extra_name}"

  # 이 프로세스는 지정한 GPU만 보이도록
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  
  # --- Step 1
  python train.py -s "${source_path}" \
    -m "${model_path}" \
    --eval --resolution 1352 --iterations 6000 --opacity_reset_interval 200000\
    --disable_viewer \
    --test_iterations ${ITERS[@]} \
    --save_iterations ${ITERS[@]} \
    --freetimegs_path ${freetimegs_path} \
    --frame_num "${frame_num}" \
    --densification_interval 2000 \
    # --position_lr_init 0.0000016 \
    # --position_lr_init 0.00008 \
    # --feature_lr 0.00125 \
    # --opacity_lr 0.0125 \
    # --scaling_lr 0.0025 \
    # --rotation_lr 0.0005 \
    # --rotation_lr 0.0 
    # --position_lr_init 0.0000016 \

  # --- Step 2
  for it in "${ITERS[@]}"; do
    python render.py -m "${model_path}" --skip_train --iteration "$it"
  done

  python metrics.py -m "${model_path}"
}

for frame_num in $(seq "$start_num" "$end_num"); do
  gpu_id=$(( frame_num % gpu_ea ))

  # 동시 실행 개수(gpu_ea) 초과 시 대기
  while [ "$(jobs -pr | wc -l)" -ge "$gpu_ea" ]; do
    sleep 1
  done

  # 로그 분리(선택)
  logdir="${output_path}/${dataset_name}/logs"
  mkdir -p "$logdir"
  run_one "$frame_num" "$gpu_id" >"${logdir}/f${frame_num}.log" 2>&1 & # 비동기 호출
  # run_one "$frame_num" "$gpu_id"
  echo "Launched frame ${frame_num} on GPU ${gpu_id}"
done

wait
cd $current_path
echo "All done."
