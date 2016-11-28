if [ "$#" -ne 2 ]; then
    echo "Usage: exec <number of times to concatenate> <output file>";
fi

MAX=$1
FILE=$2
WIDTH=640
HEIGHT=$((480 * MAX))

printf "P6\n$WIDTH $HEIGHT\n255\n" > $FILE

for i in `seq 1 $MAX`;
do
    cat ./images/valve.raw.pbm >> $FILE
done
