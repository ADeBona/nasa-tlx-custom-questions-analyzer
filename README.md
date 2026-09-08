# NASA-TLX

Simplistic electronic version of the "NASA Task Load Index" without subscale weighting (often referred to as Raw TLX) by Hart and Staveland, extended with custom Likert items and a group-level analysis script. Related literature:

 * Hart, S. G., & Staveland, L. E. (1988). Development of NASA-TLX (Task Load Index): Results of empirical and theoretical research. In Advances in psychology (Vol. 52, pp. 139-183). North-Holland.
 * Hart, S. G. (2006, October). NASA-task load index (NASA-TLX); 20 years later. In Proceedings of the human factors and ergonomics society annual meeting (Vol. 50, No. 9, pp. 904-908). Sage CA: Los Angeles, CA: Sage Publications.

Based on the original [nasa-tlx](https://github.com/TimDomino/nasa-tlx) GUI by Tim Weißker.

## Dependencies
 * Python 3 with Tkinter (`sudo apt-get install python3-tk` on Debian-based systems; bundled with the standard installer on macOS/Windows) - http://www.python.org/
 * For `analyze_results.py`: `pip3 install -r requirements.txt`

## Usage Instructions

### Collecting responses
Run the test with `python3 nasa-tlx.py`. The experiment and condition labels, as well as the custom Likert items, can be altered at the top of `nasa-tlx.py`. Each question is rated on a scale from 0 to 100 in steps of 5, but the numeric values are hidden to the participant. When pressing the submit button, the results are appended to `nasa-tlx-results.csv`, with one row per participant x condition.

### Analyzing results
Run `python3 analyze_results.py [path/to/nasa-tlx-results.csv]` to generate:
 1. Descriptive statistics (n, mean, SD, SEM, 95% CI, median, min, max) for the six raw NASA-TLX subscales, the unweighted overall Raw-TLX score, and every custom Likert item, broken down by condition.
 2. Within-subject paired comparisons between conditions (paired t-test, Wilcoxon signed-rank test, Cohen's dz) for every measure above.
 3. Analysis of the free-text feedback comment.
 4. Publication-ready figures (PNG, 300 dpi) written to `./analysis_output/`.
 5. A Markdown summary (`analysis_output/results_summary.md`) with every number pre-formatted for direct use in a paper.

`scipy` is optional and enables the paired-comparison tests and 95% CIs; without it those sections are skipped with a warning instead of crashing.
