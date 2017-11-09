#!/bin/bash

### grep from dat files and save in CSV style format
NAME="*N_1500.dat"
#NAME="*.dat"

K_LINE=`find . -name "$NAME" -exec  grep 'k is:' {} \;`
K=`echo "$K_LINE" | grep -Eo '[0-9]{1,7}'`

METHOD_LINE=`find . -name "$NAME" -exec  grep 'The argument supplied for the method is:' {} \;`
METHOD=`echo "$METHOD_LINE" | awk '{print $8}'`

N_LINE=`find . -name "$NAME" -exec  grep 'The argument supplied for N is:' {} \;`
N=`echo "$N_LINE" | awk '{print $7}'`

THREAD_LINE=`find . -name "$NAME" -exec  grep 'THREADS:' {} \;`
THREAD=`echo "$THREAD_LINE" | awk '{print $3}'`

TIME_LINE=`find . -name "$NAME" -exec  grep 'elapsed' {} \;`
TIME_REP=`echo "$TIME_LINE" | awk '{print $3}'`
TIME="$(echo $TIME_REP | sed 's/elapsed/\n/g')"

paste <(echo "$K") <(echo "$N") <(echo "$THREAD") <(echo "$TIME") <(echo "$METHOD") --delimiters ';'
#paste <(echo "$METHOD") <(echo "$THREAD") <(echo "$N") <(echo "$TIME") <(echo "$K") --delimiters ';' | sort