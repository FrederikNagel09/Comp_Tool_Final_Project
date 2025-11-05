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

- Use confusion matrix to showcase accuracy of classification



## Datasets: 
- https://www.kaggle.com/datasets/navjotkaushal/human-vs-ai-generated-essays/data
- https://www.kaggle.com/datasets/pratyushpuri/ai-vs-human-content-detection-1000-record-in-2025
- https://www.kaggle.com/datasets/denvermagtibay/ai-generated-essays-dataset

This is the big dataset
- https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text

- We could use the big + one other to train and the others to validate? 
- We do need to check that none of the text samples across all datasets are identical.
- Look at code section in kaggle of each dataset and find inspiration of what others have done
