#!/bin/bash
VID_FOLDER=$1
CLEAR_ITEM=$2

cd $VID_FOLDER #can be chnaged

if [ "$CLEAR_ITEM" == "critic" ]; then
    find . -type f -name "*critic_assessment.json" -delete
    find . -type f -name "*critic_assessment_reformatted.json" -delete
    find . -type f -name "*critic_model.json" -delete
    find . -type f -name "*critic_response.json" -delete

elif [ "$CLEAR_ITEM" == "re_evaluated" ]; then
    find . -type f -name "*re_evaluated.json" -delete

elif [ "$CLEAR_ITEM" == "answers" ]; then
    find . -type f -name "*answers.json" -delete
    find . -type f -name "*answers_reformatted.json" -delete
    find . -type f -name "*os_model.json" -delete

elif [ "$CLEAR_ITEM" == "all" ]; then
    find . -type f -name "*critic_assessment.json" -delete
    find . -type f -name "*critic_assessment_reformatted.json" -delete
    find . -type f -name "*critic_model.json" -delete
    find . -type f -name "*critic_response.json" -delete
    find . -type f -name "*re_evaluated.json" -delete
    find . -type f -name "*answers.json" -delete
    find . -type f -name "*answers_reformatted.json" -delete
    find . -type f -name "*os_model.json" -delete

fi