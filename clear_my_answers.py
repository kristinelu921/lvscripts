import os
import argparse

parser = argparse.ArgumentParser(
    prog = 'clear_my_answers.py',
    description = 'clears my answers from video sets',
    epilog = 'do that'
)

parser.add_argument('dirname')
parser.add_argument('file_type', choices=['answers', 'answers_reformatted', 'critic_assessment', 're_evaluated'])
args = parser.parse_args()


dir = args.dirname
nums = os.listdir(dir)

for num in nums[:10]:
    print(num)
    answers_path = f"./{dir}/{num}/{num}_answers.json"
    answers_reformatted_path = f"./{dir}/{num}/{num}_answers_reformatted.json"
    critic_path = f"./{dir}/{num}/{num}_critic_assessment.json"
    re_evaluated_path = f"./{dir}/{num}/{num}_re_evaluated.json"
    
    
    if os.path.exists(f"./{dir}/{num}/{num}_{args.file_type}.json"):
        #os.remove(f"./{dir}/{num}/{num}_{args.file_type}.json")
        #print(f"Deleted: ./{dir}/{num}/{num}_{args.file_type}.json")
        pass
        
    
    if os.path.exists(answers_path):
        os.remove(answers_path)
        print(f"Deleted: {answers_path}")
        pass
    
    if os.path.exists(answers_reformatted_path):
        os.remove(answers_reformatted_path)
        print(f"Deleted: {answers_reformatted_path}")
        pass

    if os.path.exists(critic_path):
        os.remove(critic_path)
        print(f"Deleted: {critic_path}")
    
    if os.path.exists(re_evaluated_path):
        os.remove(re_evaluated_path)
        print(f"Deleted: {re_evaluated_path}")
    
