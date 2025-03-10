#!/bin/bash

# Initialize Git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: PPO Agent for Lux AI Season 3"

# Instructions for connecting to GitHub
echo "
Repository setup complete! 

Next steps to push to GitHub:

1. Create a new repository on GitHub:
   - Go to https://github.com/new
   - Name it 'lux-ppo-agent' (or your preferred name)
   - Do NOT initialize with README, .gitignore, or license

2. Connect your local repo to GitHub:
   git remote add origin https://github.com/YOUR_USERNAME/lux-ppo-agent.git
   git branch -M main
   git push -u origin main

3. Update the Colab notebook:
   - Edit train_on_colab.ipynb to replace 'YOUR_USERNAME' with your actual GitHub username

That's it! Your code is now on GitHub and ready for training on Colab.
"