#!/bin/bash
for i in $(find paraopt | egrep "\.pyc$|\.py~$|\.pyc~$|\.bak$$") ; do rm -v ${i}; done
rm -vr build
