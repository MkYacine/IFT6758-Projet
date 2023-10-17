# IFT6758-Projet

This repository will be a shared work space for our project on NLH data analysis.

## Using the project
Be sure to `run pip install -e .` in the project's root directory immediately after cloning.

### 1. Downloading the data
To download the NHL raw data for a specific season: In the project's root directory, run `.\src\data/scrape.py *year* *optional: output path*`  
e.g. `./src/data/scrape.py 2017` download data for the 2017/2018 season.
This will create a `*year*.json` file in the `/data/datasets/json_files` containing the raw data.
You can also specify the output path as a second arguement

### 2. Cleaning the data
To clean the data and obtain the features required for visualizations, run `/src/features/clean.py *path_to_json* *optional: output path`  
e.g. `/src/data/scrape.py /data/datasets/csv_files/2017.csv`  
If the output_path is left emtpy, the data will be written to `/data/datasets/csv_files/*year*.json`  

### 3. Visualizing the data
For data visualizations, use the notebooks found in `/notebooks`  
`Interactif_debug_tool.ipynb`: Debugging tool to view every offensive play in every season.  
`simple_visualization.ipynb`: Simple visualizations: shot efficiency by type, shot efficiency by distance, etc...  
`advanced_visualization.ipynb`: Advanced visualization: shot map for every team by season.  

## Environment setup
For uniformity, I suggest using virtualenv with pip and requirements.txt to keep the tools used up to date.  
With pip and virtualenv installed on your system, follow these steps:  
1. Clone the repository with `git clone https://github.com/MkYacine/IFT6758-Projet.git`
2. In your project repo, create the virtual environment with `vitualenv venv`
3. Activate your virtual environment with `venv\Scripts\activate`
4. Run `pip install -r requirements.txt` to install all required packages.  
The requirements.txt file should be maintained manually; before committing your changes, run `pip freeze > requirements.txt` to update.  

## General workflow
If you're inexperienced with Git, here is a general workflow that I have used for previous team projects:

### 1. Create a New Branch:
Within your IDE, create a new branch from the main branch.  
Name the branch in a descriptive manner, such as feature/feature-name, bugfix/bug-name.  
`git checkout -b feature/feature-name main`

### 2. Implement Your Feature:
Checkout to your newly created branch.  
Start implementing your feature, regularly committing changes with meaningful commit messages.  
`git add <file1> <file2> <file3>`  
`git commit -m "Implemented a new feature: feature-name"`

### 3. Fetch and Merge Main:
Before pushing your changes, fetch the latest changes from the main branch and merge them into your feature branch to resolve any conflicts and ensure smooth integration.  
`git checkout main`  
`git pull`  
`git checkout feature/feature-name`  
`git merge main`  

### 4. Push Your Branch:
Once you've resolved any conflicts and are satisfied with your changes, push your feature branch to the remote GitHub repository on your remote branch.  
`git push -u origin feature/feature-name`

### 5. Create a Pull Request:
Go to the GitHub repository online and navigate to the “Pull requests” tab.  
Click on “New pull request”.  
Set the base branch to main and the compare branch to your feature branch.

### 6. Request Reviews:
Request reviews from at least two other team members. 
Respond to any comments or requested changes.

### 7. Get Approval and Merge:
Click on “Merge pull request” to merge your changes into the main branch.  

### 8. Clean Up:
Delete the remote feature branch from GitHub.  
Switch to the main branch in your local environment, pull the latest changes, and delete the local feature branch.  
`git branch -d feature/feature-name`  
`git push origin --delete feature/feature-name`
