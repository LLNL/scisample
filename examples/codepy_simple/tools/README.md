# convert maestro examples to codepy format
tools/convert_maestro_to_codepy.py ../maestrowf/sample*yaml
#
python3 tools/make_mdpp_codepy_docs.py best_candidate column_list cross_product list random uqpipeline > codepy_docs.mdpp
markdown-pp codepy_docs.mdpp -o codepy_docs.md
