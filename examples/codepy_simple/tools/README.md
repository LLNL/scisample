# run codepy examples:
parallel -j 1 "echo {}; codepy run -c {}" :::  codepy_config_*.yaml

# cd into "studies" directory and extract output
parallel "echo {}; cd {}; more *2022*/run*/*/out.txt > ../{}_out.txt" ::: *20220724-12*

# edit names and files as needed

# make markdown document
python3 tools/make_mdpp_codepy_docs.py best_candidate column_list cross_product csv_column csv_row custom list random uqpipeline  > codepy_docs.mdpp
markdown-pp codepy_docs.mdpp -o codepy_docs.md

# edit markdown file to remove extra newlines

# first 6 examples were converted from maestro to codepy
../codepy_config_best_candidate.yaml  ../codepy_config_cross_product.yaml  ../codepy_config_random.yaml
../codepy_config_column_list.yaml     ../codepy_config_list.yaml	   ../codepy_config_uqpipeline.yaml

# convert maestro examples to codepy format
tools/convert_maestro_to_codepy.py ../maestrowf/sample*yaml

# needed to hand edit "min/max" dictionaries to a more "standard" yaml format

# the last three examples need to be converted back to maestro format
codepy_config_csv_column.yaml  codepy_config_csv_row.yaml  codepy_config_custom.yaml

# 

