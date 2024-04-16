cd .. && bash package.sh
cd deployment_train 
output_dict=$(python3 start_vertex.py)
echo $output_dict
