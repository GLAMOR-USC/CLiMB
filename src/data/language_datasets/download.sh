dataset="cosmosqa"

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

function cosmosqa {
    wget https://raw.githubusercontent.com/wilburOne/cosmosqa/master/data/train.csv
    wget https://raw.githubusercontent.com/wilburOne/cosmosqa/master/data/valid.csv
    wget https://raw.githubusercontent.com/wilburOne/cosmosqa/master/data/test.jsonl
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
elif [ ${dataset} == "cosmosqa" ]
then
   cosmosqa 
else
    echo "Not defined!"
fi

