rm out/*

# assign data path to variable data_path on shell script
data="bbcsport"
isreduced="True"

# echo "################### computing WMD distances for $data_path ##################"
python compute_distances.py --filename bbcsport --reduced $isreduced 
python summarize_distances.py --filename bbcsport --reduced $isreduced 
python compute_distances.py --filename bbcsport --reduced $isreduced --tfidf
python summarize_distances.py --filename bbcsport --reduced $isreduced --tfidf
python evaluate.py --filename bbcsport --reduced $isreduced  

# make a array of p and k values
p_values=(2)
k_values=(0.01)
isrpw="True"

for p in ${p_values[@]}
do
    for k in ${k_values[@]}
    do
        echo "################### computing RPW distances for $data_path w/ p=$p and k=$k ##################"
        python compute_distances.py --filename $data --p $p --k $k --reduced $isreduced --rpw $isrpw
        # python compute_distances.py data/$data_path.mat 1 0
        python summarize_distances.py --filename $data --reduced $isreduced --rpw $isrpw 
        python compute_distances.py --filename $data --tfidf --p $p --k $k --reduced $isreduced --rpw $isrpw
        # python compute_distances.py data/$data_path.mat 1 0 --tfidf
        python summarize_distances.py --filename $data --tfidf --reduced $isreduced --rpw $isrpw 
        python evaluate.py --filename $data --reduced $isreduced --rpw $isrpw
    done
done

