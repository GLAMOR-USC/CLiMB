dataset="commonsenseqa"

function hellaswag {
    wget https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl
    wget https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl
    wget https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl
}

function piqa {
    wget https://yonatanbisk.com/piqa/data/train.jsonl
    wget https://yonatanbisk.com/piqa/data/valid.jsonl
    wegt https://yonatanbisk.com/piqa/data/tests.jsonl
    wget https://yonatanbisk.com/piqa/data/train-labels.lst
    wget https://yonatanbisk.com/piqa/data/valid-labels.lst
}

function commonsenseqa {
    wget https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl
    wget https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl
    wget https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl
}


echo "Download ${dataset}..."
mkdir ${dataset}
cd ${dataset}

if [ ${dataset} == "hellaswag" ]
then
    hellaswag
elif [ ${dataset} == "piqa" ]
then
    piqa
elif [ ${dataset} == "commonsenseqa" ]
then
   commonsenseqa 
else
    echo "Not defined!"
fi

