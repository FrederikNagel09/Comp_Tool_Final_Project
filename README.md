# Comp_Tool_Final_Project

## Follow these instructions to set up project
1) make sure your directory is this project folder
run in terminal
pwd
should look like .../Comp_Tool_Final_Project

2) Initialize virtual environment:
run in terminal 
python3 -m venv .venv
source .venv/bin/activate

3) Install and sync project dependencies
run in terminal 
pip install uv
uv pip sync


## Makefile instructions
run in terminal:
make check
- This runs both ruff format and ruff check, automatically formatting code to industry standards and giving warnings where changes should be made to follow proper formatting and good coding etiquette.



## Project Ideas: 
- TF-IDF → LogisticRegression
    - Could be graph with k-means classification instead of logisticregression
    - Could be more powerfull DL model than logistic regression
    - TF-IDF: teat every text sample as a "document" and use it to create an embedding vector for each text

- Use other embedding method + some classification method





