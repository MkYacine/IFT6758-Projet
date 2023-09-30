# IFT6758-Projet

This repository will be a shared work space for our project on NLH data analysis.

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
`git add .`  
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
Fill out the pull request template with all relevant information, describing your feature, the problem it solves, and any additional context or screenshots.  

### 6. Request Reviews:
Request reviews from at least two other team members (or as per your team’s collaboration policy).  
Respond to any comments or requested changes promptly and make necessary modifications.  

### 7. Get Approval and Merge:
Address any feedback from your team members and make requested changes.  
Once your PR receives approvals from the required number of team members, and all discussions are resolved, it is ready to be merged.  
Click on “Merge pull request” to merge your changes into the main branch.  

### 8. Clean Up:
Once your feature branch is merged, it’s good practice to delete the remote feature branch from GitHub.  
Additionally, switch to the main branch in your local environment, pull the latest changes, and delete the local feature branch.  
`git branch -d feature/feature-name`  
`git push origin --delete feature/feature-name`
