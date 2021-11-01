FILE=$1
TYPE=$2

ncdump ${FILE} > tmp_dump.txt
sed '/.*pr =.*/{s///;q;}' tmp_dump.txt > structure.txt

rm tmp_dump.txt
