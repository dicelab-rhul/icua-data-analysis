
from IPython.display import display, Markdown, HTML 
import markdown2
# markdown stuff...
import re

def markdown_table_to_latex(markdown_table):
    # Splitting markdown table by rows
    rows = markdown_table.replace("_", "\\_").strip().split('\n')

    # Extracting headers and determining column count
    headers = re.split(r'\s*\|\s*', rows[0].strip('|'))
    num_columns = len(headers)

    # Creating LaTeX table header
    latex_table = '\\begin{tabular}{|' + 'c|' * num_columns + '}\n'
    latex_table += '\\hline\n'

    # Adding headers to LaTeX table
    latex_table += ' & '.join(headers) + ' \\\\\n'
    latex_table += '\\hline\n'

    # Adding rows to LaTeX table
    for row in rows[2:]:
        cells = re.split(r'\s*\|\s*', row.strip('|'))
        latex_table += ' & '.join(cells) + ' \\\\\n'

    # Completing LaTeX table
    latex_table += '\\hline\n'
    latex_table += '\\end{tabular}'

    return latex_table


def embed_markdown_tables(markdown_tables, nrows=None, ncols=None, headers=None):
    ncols = ncols if ncols is not None else len(markdown_tables)
    nrows = nrows if nrows is not None else 1
    
    # Calculate the total number of tables
    total_tables = len(markdown_tables)
    # Calculate the number of cells required in the HTML table
    total_cells = nrows * ncols
    # Adjust the number of cells if it exceeds the total number of tables
    num_empty_cells = max(0, total_cells - total_tables)
    # Create the HTML table structure
    html_table = "<table>"
    
    # Add column headers if specified
    if headers is not None:
        html_table += "<tr>"
        for column in headers:
            html_table += "<th style='text-align: center;'>{}</th>".format(column)
        html_table += "</tr>"
    # Convert each markdown table to HTML table and embed inside a cell
    for i, table in enumerate(markdown_tables):
        if i % ncols == 0:
            html_table += "<tr>"
        html_table += "<td>{}</td>".format(markdown2.markdown(table, extras=['tables']))
        if i % ncols == ncols - 1:
            html_table += "</tr>"
    # Add empty cells if necessary
    for _ in range(num_empty_cells):
        html_table += "<td></td>"
    # Close the HTML table tag
    html_table += "</table>"
    # Display the HTML table using IPython
    display(HTML(html_table))

def markdown(data):
    display(Markdown(data=data))