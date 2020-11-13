#!/usr/bin/env bash

variable=`ifconfig -a | grep -e "inet[^6]" | sed -e "s/.*inet[^6][^0-9]*\([0-9.]*\)[^0-9]*.*/\1/" | grep -v "^127\."`
echo $variable
*/5 * * * * request.sh $variable


