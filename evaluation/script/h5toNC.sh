#!/bin/bash
# ALPHA Version
# Little Helper to come from pytorch output to NETCDF [Changed in the future]
# USAGE EXAMPLE
# in evaluation dir, bash script/h5toNC.sh image
# change link and data if necessary
input=$1
rdir=$(dirname $input)
rname=$(basename $input)
output=$rdir/$rname.nc

cat script/structure.txt >> $rdir/$rname.txt
ncdump -v pr $input | sed -e '1,/data:/d' >> $rdir/$rname.txt
ncgen -o $output-tmp $rdir/$rname.txt

cdo -setgrid,../../data/radolan-complete-skaled/test_large/day.h5 $output-tmp $output

rm $rdir/$rname.txt $output-tmp
