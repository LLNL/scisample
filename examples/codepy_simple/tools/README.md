# convert maestro examples to codepy format
tools/convert_maestro_to_codepy.py ../maestrowf/sample*yaml

# need to hand edit "min/max" dictionaries to a more "standard" yaml format

# run codepy examples:
parallel -j 1 "echo {}; codepy run -c {}" :::  *config*yaml

# make markdown document
python3 tools/make_mdpp_codepy_docs.py best_candidate column_list cross_product list random uqpipeline > codepy_docs.mdpp
markdown-pp codepy_docs.mdpp -o codepy_docs.md
