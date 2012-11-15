#!/bin/bash
echo Cleaning python code in \'`pwd`\' and subdirectories
for file in $(find paraopt *.py | egrep "(\.py$)"); do
  echo Cleaning ${file}
  sed -i -e $'s/\t/    /g' ${file}
  sed -i -e $'s/[ \t]\+$//g' ${file}
  sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' ${file}
done
