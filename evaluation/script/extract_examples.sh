# Extract specific images from test data set
FILE=$1
date_1=2017-01-12T23
date_2=2017-04-17T15
date_3=2017-05-02T12
date_4=2017-05-13T12
date_5=2017-06-04T03
date_6=2017-06-29T16
date_7=2017-07-12T14
date_8=2017-09-02T13
cdo select,date=$date_1 ${FILE} ${FILE}tmp1.nc
cdo select,date=$date_2 ${FILE} ${FILE}tmp2.nc
cdo select,date=$date_3 ${FILE} ${FILE}tmp3.nc
cdo select,date=$date_4 ${FILE} ${FILE}tmp4.nc
cdo select,date=$date_5 ${FILE} ${FILE}tmp5.nc
cdo select,date=$date_6 ${FILE} ${FILE}tmp6.nc
cdo select,date=$date_7 ${FILE} ${FILE}tmp7.nc
cdo select,date=$date_8 ${FILE} ${FILE}tmp8.nc
cdo mergetime ${FILE}tmp* ${FILE}_selected
rm ${FILE}tmp*
