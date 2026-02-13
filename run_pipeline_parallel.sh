#!/bin/bash
# Run test_pipeline on all 4 chunks in parallel
# Usage: ./run_pipeline_parallel.sh <base_videos_dir> [additional args]
# Example: ./run_pipeline_parallel.sh /mnt/ssd/data/longvideobench/videos_processed_val --mode full

if [ $# -lt 1 ]; then
    echo "Usage: $0 <base_videos_dir> [additional args]"
    echo ""
    echo "Examples:"
    echo "  $0 /mnt/ssd/data/longvideobench/videos_processed_val --mode full"
    echo "  $0 /mnt/ssd/data/videomme/video_mme_long/videos_processed --mode full --pass-all-subtitles-to-llm"
    echo "  $0 /mnt/ssd/data/lvbench/videos_processed --mode full"
    exit 1
fi

BASE_VIDEOS_DIR="$1"
shift  # Remove first argument, rest are extra args
EXTRA_ARGS="$@"

# Extract base name for output directories and log files
BASE_NAME=$(basename "$BASE_VIDEOS_DIR")

echo "Starting parallel pipeline runs on 4 chunks..."
echo "Base directory: ${BASE_VIDEOS_DIR}"
echo "Extra args: ${EXTRA_ARGS}"
echo ""

# Run each chunk in the background
for i in 1 2 3 4; do
    CHUNK_DIR="${BASE_VIDEOS_DIR}_chunk${i}"
    OUTPUT_DIR="results_${BASE_NAME}_chunk${i}"
    LOG_FILE="${BASE_NAME}_chunk${i}_pipeline.log"
    PID_FILE="${BASE_NAME}_pipeline_pids.txt"

    echo "Starting Chunk $i: ${CHUNK_DIR}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "  Log: ${LOG_FILE}"
    echo "  Command: python test_pipeline.py ${CHUNK_DIR} --output-dir ${OUTPUT_DIR} ${EXTRA_ARGS}"

    # Run in background and redirect output to log file
    nohup python test_pipeline.py "${CHUNK_DIR}" --output-dir "${OUTPUT_DIR}" ${EXTRA_ARGS} \
        > "${LOG_FILE}" 2>&1 &

    PID=$!
    echo "  PID: $PID"
    echo ""

    # Store PID for later
    echo $PID >> "${PID_FILE}"
done

echo "✅ All 4 chunks started in background!"
echo ""
echo "Monitor progress:"
for i in 1 2 3 4; do
    echo "  tail -f ${BASE_NAME}_chunk${i}_pipeline.log"
done
echo ""
echo "Check running processes:"
echo "  ps aux | grep test_pipeline"
echo ""
echo "Wait for all to complete:"
echo "  wait \$(cat ${BASE_NAME}_pipeline_pids.txt)"
