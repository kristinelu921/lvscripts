#!/bin/bash

RUN_DATE=$1
VIDEO_DIR=$2
CAPTIONS=$3

ARCHIVE_DIR="/mnt/ssd/data/run_archive/run_${RUN_DATE}"

# Create archive directory if it doesn't exist
mkdir -p "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR/results"
cd $VIDEO_DIR

echo "Archiving JSON files to: $ARCHIVE_DIR"
echo ""

# Count and move various JSON files to archive
#TODO: USE -o to make this concise.
MODEL_COUNT=$(find . -type f -name "*_model.json*" | wc -l)
echo "  Found $MODEL_COUNT files"
find . -type f -name "*_model.json*" -exec mv {} "$ARCHIVE_DIR/" \;

RESPONSE_COUNT=$(find . -type f -name "*_response.json*" | wc -l)
echo "  Found $RESPONSE_COUNT files"
find . -type f -name "*_response.json*" -exec mv {} "$ARCHIVE_DIR/" \;

ANSWERS_COUNT=$(find . -type f -name "*_answers.json*" | wc -l)
echo "  Found $ANSWERS_COUNT files"
find . -type f -name "*_answers.json*" -exec mv {} "$ARCHIVE_DIR/" \;

REFORMATTED_COUNT=$(find . -type f -name "*_answers_reformatted.json*" | wc -l)
echo "  Found $REFORMATTED_COUNT files"
find . -type f -name "*_answers_reformatted.json*" -exec mv {} "$ARCHIVE_DIR/" \;

CRITIC_COUNT=$(find . -type f -name "*_critic_assessment.json*" | wc -l)
echo "  Found $CRITIC_COUNT files"
find . -type f -name "*_critic_assessment.json*" -exec mv {} "$ARCHIVE_DIR/" \;

RE_EVALUATED_COUNT=$(find . -type f -name "*_re_evaluated.json*" | wc -l)
echo "  Found $RE_EVALUATED_COUNT files"
find . -type f -name "*_re_evaluated.json*" -exec mv {} "$ARCHIVE_DIR/" \;

TOTAL=$((MODEL_COUNT + RESPONSE_COUNT + ANSWERS_COUNT + REFORMATTED_COUNT + CRITIC_COUNT + RE_EVALUATED_COUNT))
echo ""
echo "Total files moved: $TOTAL"
echo "Files archived to $ARCHIVE_DIR"

cd /mnt/ssd/data/lvscripts/${RUN_DATE}
find . -type f -name "test_*" -exec mv {} "$ARCHIVE_DIR/results/" \;
find . -type f -name "simplified_results_*.json" -exec mv {} "$ARCHIVE_DIR/results/" \;

echo "Archiving test report files to $ARCHIVE_DIR/results/"

cd /mnt/ssd/data/critic_analysis
find . -type f -name "*.png" -exec mv {} "$ARCHIVE_DIR/results/" \;
echo "Archiving critic analysis files to $ARCHIVE_DIR/results/"

if [ -n "$CAPTIONS" ]; then
    echo "Archiving captions to $ARCHIVE_DIR/captions"
    NUM_CAPTIONS=$(find . -type f \( -name "*_captions*.json*" -o -name "*_logs.txt" -o -name "*_summary.txt" \) | wc -l)
    echo "  Found $NUM_CAPTIONS files"
    mkdir -p "$ARCHIVE_DIR/captions"
    find . -type f \( -name "*_captions*.json*" -o -name "*_logs.txt" -o -name "*_summary.txt" \) -exec mv {} "$ARCHIVE_DIR/captions/" \;
fi

echo "Done!"
