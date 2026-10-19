# Data Science Internship Project: Health and Wealth of Nations

A five-week data science project covering data preparation, visual storytelling, hypothesis testing,
machine learning and strategic reporting. All scripts download their public data directly, so no local
data files are needed.

| Week | Topic | Dataset | Script | Report |
|------|-------|---------|--------|--------|
| 1 | Data acquisition, cleaning and EDA | Titanic passengers (891 rows) | `week1/analysis.py` | `reports/Week1_Data_Cleaning_EDA_Report.docx` |
| 2 | Advanced visualization and storytelling | Gapminder (142 countries, 1952-2007) | `week2/story.py` | `reports/Week2_Data_Storytelling_Report.docx` |
| 3 | Statistical analysis and hypothesis testing | Gapminder | `week3/hypothesis_tests.py` | `reports/Week3_Hypothesis_Testing_Report.docx` |
| 4 | Machine learning model development and evaluation | Gapminder | `week4/ml_model.py` | `reports/Week4_ML_Model_Report.docx` |
| 5 | Final project report and recommendations | All of the above | (none, synthesis) | `reports/Week5_Final_Project_Report.docx` |

## How to run

```bash
pip install -r requirements.txt
cd week1 && python analysis.py      # cleaning, EDA, figures, titanic_clean.csv
cd ../week2 && python story.py      # six storytelling visualizations
cd ../week3 && python hypothesis_tests.py   # five hypothesis tests
cd ../week4 && python ml_model.py   # classification models and evaluation
```

Each script writes its figures (PNG) and, where relevant, a results JSON file into its own folder.
Python 3.10 or newer is recommended. An internet connection is required the first time to download the datasets.

## Data sources
- Titanic: `datasciencedojo/datasets` on GitHub
- Gapminder: `plotly/datasets` on GitHub

## Key results
- Population-weighted life expectancy rose from 48.9 to 68.9 years between 1952 and 2007.
- Continent explains about 63% of the variation in 2007 life expectancy (ANOVA, eta-squared 0.63).
- Logistic regression predicts whether a country has life expectancy of 70+ with 90% accuracy and 0.96 AUC on unseen countries.
