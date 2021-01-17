#!/bin/bash

script=$0
file=$1
echo "running $0"

# preserve files
#rm_files=\
#VRP100_epoch2.pt\
#VRP100_epoch3.pt\
#VRP100_epoch4.pt\
#VRP100_epoch5.pt\


#rm -rf ${rm_files}





# preserve files
file=
rm_files=
cp ${file} ../

# remove
rm -rf ${rm_files}

mv ../${file} .
