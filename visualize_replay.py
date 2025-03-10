#!/usr/bin/env python
"""
Utility script to visualize replay files using the Lux-Eye visualizer.
"""
import os
import argparse
import webbrowser
import glob
import json
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Lux AI S3 replay files")
    parser.add_argument("--replay", type=str, default="", help="Path to replay file (.json)")
    parser.add_argument("--replay-dir", type=str, default="", help="Directory containing replay files")
    parser.add_argument("--html", action="store_true", help="Convert JSON replay to HTML for web browser viewing")
    parser.add_argument("--output-dir", type=str, default="html_replays", help="Directory to save HTML replays")
    return parser.parse_args()

def json_to_html(json_path, output_dir="html_replays"):
    """Convert a JSON replay to an HTML file for browser viewing."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the replay
    with open(json_path, 'r') as f:
        replay = json.load(f)
    
    # Generate HTML filename from JSON filename
    json_filename = os.path.basename(json_path)
    html_filename = os.path.splitext(json_filename)[0] + ".html"
    html_path = os.path.join(output_dir, html_filename)
    
    # Create HTML content with embedded replay data
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="https://s3vis.lux-ai.org/eye.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />

    <title>Lux Eye S3 - {json_filename}</title>

    <script>
window.episode = {json.dumps(replay)};
    </script>

    <script type="module" crossorigin src="https://s3vis.lux-ai.org/index.js"></script>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
"""
    
    # Write the HTML file
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created HTML replay at: {html_path}")
    return html_path

def main():
    args = parse_args()
    
    # Check if we have a replay file or directory
    replay_files = []
    
    if args.replay:
        if os.path.exists(args.replay):
            replay_files.append(args.replay)
        else:
            print(f"Error: Replay file {args.replay} not found.")
            return
    
    elif args.replay_dir:
        if os.path.isdir(args.replay_dir):
            # Find all JSON replay files in the directory
            replay_files = glob.glob(os.path.join(args.replay_dir, "*.json"))
            if not replay_files:
                print(f"No JSON replay files found in {args.replay_dir}")
                return
        else:
            print(f"Error: Replay directory {args.replay_dir} not found.")
            return
    
    else:
        print("Please specify either --replay or --replay-dir")
        return
    
    # Convert and open replays
    for replay_file in replay_files:
        if args.html:
            # Convert JSON to HTML
            html_path = json_to_html(replay_file, args.output_dir)
            
            # Open in browser
            print(f"Opening {html_path} in browser...")
            webbrowser.open(f"file://{os.path.abspath(html_path)}")
            
            # Wait a bit if we're opening multiple files
            if len(replay_files) > 1:
                time.sleep(1)
        else:
            # Use Lux-Eye visualizer (must be running)
            print(f"To visualize {replay_file}, use the Lux-Eye web app:")
            print(f"1. Open https://s3vis.lux-ai.org/")
            print(f"2. Upload the replay file: {os.path.abspath(replay_file)}")
            print("\nOr run this script with --html to convert to HTML and open directly.\n")
            
if __name__ == "__main__":
    main()