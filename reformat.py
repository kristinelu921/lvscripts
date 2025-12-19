import json
import ast
import os
import argparse

parser = argparse.ArgumentParser(
    prog = "reformat.py",
    description = "reofmrats json files"
)

parser.add_argument('dirname')
args = parser.parse_args()

dir = args.dirname
def reformat_answers(dir):
    # Read the malformed JSON file
    for num in os.listdir(f'./{dir}'):
        if os.path.exists(f'./{dir}/{num}/{num}_answers_reformatted.json'):
            with open(f'./{dir}/{num}/{num}_answers_reformatted.json', 'r') as f:
                content = f.read()

            # Check if it's a JSON string (starts with quote)
            if content.startswith('"'):
                try:
                    # Try to evaluate as a Python literal string
                    content_unescaped = ast.literal_eval(content + '"')  # Add closing quote if truncated
                except:
                    # If that fails, try without adding quote
                    try:
                        content_unescaped = ast.literal_eval(content)
                    except:
                        # Last resort - manually clean it
                        if content.endswith('"'):
                            content_unescaped = content[1:-1]
                        else:
                            content_unescaped = content[1:]
                        # Replace escape sequences
                        content_unescaped = content_unescaped.replace('\\n', '\n')
                        content_unescaped = content_unescaped.replace('\\"', '"')
                        content_unescaped = content_unescaped.replace('\\/', '/')
                
                # The content is incomplete, let's try to fix it
                if not content_unescaped.rstrip().endswith(']'):
                    # Find the last complete item
                    last_complete = content_unescaped.rfind('},')
                    if last_complete > 0:
                        content_unescaped = content_unescaped[:last_complete+1] + '\n]'
                
                # Parse the JSON
                data = json.loads(content_unescaped)
            else:
                # Already proper JSON
                data = json.loads(content)

            # Write it back as proper JSON
            with open(f'./{dir}/{num}/{num}_answers_reformatted.json', 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"Successfully reformatted! Found {len(data)} questions.")
            print(num)

if __name__ == "__main__":
    reformat_answers(dir)