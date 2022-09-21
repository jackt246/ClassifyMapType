import requests
import os

#x = requests.get('https://www.ebi.ac.uk/emdb/api/search/structure_determination_method:"subtomogramAveraging"')

x = []
x = os.system("curl -X 'GET' \
  'https://www.ebi.ac.uk/emdb/api/search/structure_determination_method%3A%22tomography%22?fl=emdb_id' \
  -H 'accept: text/csv' \
  -H 'X-CSRFToken: vBP7tMPjeu8moWmB1vvXTtezPuXCFAbNZsZBELArj1ps3nLBJlEod59OmP6ljCcw' \
  -H 'Cache-Control: no-cache' \
  -H 'Pragma: no-cache' > tomograms.txt")

