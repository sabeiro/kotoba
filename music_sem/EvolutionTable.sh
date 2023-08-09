cd midi
for i in *midi ; 
do 
	../CalcEntropy.py $i 
	mv EntropyEvolution.dat ../Evolution/Ev${i/'.midi'/}.dat
done
