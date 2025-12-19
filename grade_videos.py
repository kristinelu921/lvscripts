import os
import json
import argparse

parser= argparse.ArgumentParser(
    prog = 'grade_videos.py',
    description = "grades answers from this video folder",
)

parser.add_argument('dirname')
args = parser.parse_args()


vid_folder = args.dirname
vid_paths = os.listdir(vid_folder)

not_completed_vids = []
open('graded_stats', 'w').close()
running_total = 0
running_correct = 0
running_done = 0

for num in vid_paths:

    print(num + "reached")
    my_a_path = f'{vid_folder}/{num}/{num}_answers_reformatted.json'
    ref_a_path = f'{vid_folder}/{num}/{num}_question_answers.json'

    if not os.path.exists(my_a_path):
        print(my_a_path)
        print(num)
        not_completed_vids.append(num)
        continue

    with open(my_a_path, "r") as f:
        my_answers = json.load(f)
    
    with open(ref_a_path, "r") as f:
        ref_answers = json.load(f)

    num_correct = 0
    running_done += len(my_answers)
    num_questions = len(ref_answers)
    running_total += num_questions
    correct_uids = []
    for ref_ans in ref_answers:
        for my_ans in my_answers:
            print(my_ans)
            if ref_ans["uid"] == my_ans["uid"]:
                if my_ans["answer"].upper() == ref_ans["answer"]:
                    num_correct += 1
                    running_correct += 1
                    correct_uids.append(ref_ans["uid"])
    
    with open('graded_stats', 'a') as f:
        f.write(f'{num}: {num_correct}/{num_questions}: {num_correct/num_questions}: \n Correct UIDs: {correct_uids} \n')

with open('graded_stats', 'a') as f:
    f.write("="*60 + "\n\n")
    f.write(f'TOTAL STATS: {running_correct}/{running_total} : {running_correct/running_total}')
    f.write(f'DONE STATS: {running_correct}/{running_done} : {running_correct/running_done}')

print(not_completed_vids)