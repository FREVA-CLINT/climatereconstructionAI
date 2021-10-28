#!/bin/bash

data_root=$1

mkdir $data_root/evaluations

# clean output_comp
cdo gec,0.0 $data_root/output_comp.nc $data_root/tmp.nc
cdo mul $data_root/output_comp.nc $data_root/tmp.nc $data_root/output_comp_cleaned.nc
rm tmp.nc

bash script/extract_examples.sh $data_root/image.nc
bash script/extract_examples.sh $data_root/gt.nc
bash script/extract_examples.sh $data_root/output_comp_cleaned.nc

python script/evaluate.py --data-root=$data_root --file=image.nc_selected --mask=$data_root/single_radar_fail.h5 --var=pr
python script/evaluate.py --data-root=$data_root --file=gt.nc_selected --var=pr
python script/evaluate.py --data-root=$data_root --file=output_comp_cleaned.nc_selected --var=pr

rm $data_root/*_selected

# get difference
#cdo sub $data_root/output_comp_cleaned.nc $data_root/gt.nc $data_root/sub.nc

# ectract infilled part
cdo ifnotthen $data_root/single_radar_fail.h5 $data_root/output_comp_cleaned.nc $data_root/infilled_output_comp.nc
cdo ifnotthen $data_root/single_radar_fail.h5 $data_root/gt.nc $data_root/infilled_gt.nc

# get correlation
cdo timcor -hourmean -fldmean $data_root/infilled_output_comp.nc -hourmean -fldmean $data_root/infilled_gt.nc $data_root/evaluations/timcor.nc

# get sum in field
cdo timcor -hourmean -fldsum $data_root/infilled_output_comp.nc -hourmean -fldsum $data_root/infilled_gt.nc $data_root/evaluations/fldsum_timcor.nc

# get mse
cdo sqrt -timmean -sqr -sub -hourmean -fldmean $data_root/infilled_output_comp.nc -hourmean -fldmean $data_root/infilled_gt.nc $data_root/evaluations/mse.nc

# get total fldsum
cdo fldsum -timsum $data_root/infilled_gt.nc $data_root/evaluations/fldsum_gt.nc
cdo fldsum -timsum $data_root/infilled_output_comp.nc $data_root/evaluations/fldsum_output_comp.nc

# time series of time correlation
cdo fldcor -setmisstoc,0 $data_root/infilled_output_comp.nc - setmisstoc,0 $data_root/infilled_gt.nc $data_root/evaluations/time_series.nc

# save min max mean
cdo fldmax $data_root/infilled_gt.nc $data_root/evaluations/gt_max.nc
cdo fldmax $data_root/infilled_output_comp.nc $data_root/evaluations/output_comp_max.nc
cdo fldmin $data_root/infilled_gt.nc $data_root/evaluations/gt_min.nc
cdo fldmin $data_root/infilled_output_comp.nc $data_root/evaluations/output_comp_min.nc
cdo fldmean $data_root/infilled_gt.nc $data_root/evaluations/gt_mean.nc
cdo fldmean $data_root/infilled_output_comp.nc $data_root/evaluations/output_comp_mean.nc

python script/extract_info.py --data-root=$data_root/evaluations/