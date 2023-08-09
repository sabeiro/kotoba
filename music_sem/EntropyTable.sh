echo ""> EntropyTable.dat
for i in midi/*mid?
do
	FILE=${i/'midi/'/}
	FILE=${FILE/'.midi'/}
	echo $FILE
	ENTROPY=$(./CalcEntropy.py $i 2> EntropyTable.log)
	echo $FILE $ENTROPY  >> EntropyTable.dat
done
python DistMat.py
