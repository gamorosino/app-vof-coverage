#! /bin/bash

#########################################################################################################################
#########################################################################################################################
###################                                                                                   ###################
###################title:              Files library                                           ###################
###################                                                                                   ###################
###################description:Library of functions for files management                         ###################
###################                                                                                   ###################
###################version:0.7.7                                                                 ###################
###################notes:        .                                                                 ###################
###################bash version:   tested on GNU bash, version 4.2.53                                ###################
###################                                                                                   ###################
###################autor: gamorosino                                                                 ###################
###################     email: g.amorosino@gmail.com                                                  ###################
###################                                                                                   ###################
#########################################################################################################################
#########################################################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"
STRlib=${SCRIPT_DIR}"/STRlib.sh"
source ${STRlib}

exists () {
if [ $# -lt 1 ]; then
    echo $0: "usage: exists <filename> "
    return 1;
fi

if [ -d "${1}" ]; then
echo 1;
else
([ -e "${1}" ] && [ -f "${1}" ]) && { echo 1; } || { echo 0; }
fi
};

fbasename () {
                echo `basename $1 | cut -d '.' -f 1`
};

fextension () {
                local filename=$( basename $1 )
                local extension="${filename##*.}"
echo $extension
};

remove_ext () {
local lst="";
for fn in $@ ; do
local f=`echo "$fn" | sed 's/\.hdr\.gz$//' | sed 's/\.img\.gz$//' | sed 's/\.hdr$//' | sed 's/\.img$//' | sed 's/\.nii.gz$//' | sed 's/\.nii$//' | sed 's/\.mnc.gz$//' | sed 's/\.mnc$//' | sed 's/\.$//'`;
local f=`echo "$f" | sed 's/\.hdr\.gz[ ]/ /g' | sed 's/\.img\.gz[ ]/ /g' | sed 's/\.hdr[ ]/ /g' | sed 's/\.img[ ]/ /g' | sed 's/\.nii\.gz[ ]/ /g' | sed 's/\.nii[ ]/ /g' | sed 's/\.mnc\.gz[ ]/ /g' | sed 's/\.mnc[ ]/ /g' |sed 's/\.[ ]/ /g'`;
local lst="$lst $f";
done
echo $lst;
}

checkAbsPath () {
if [ $# -lt 1 ]; then
    echo $0: usage: "checkAbsPath <file>  "
    return 1;
fi

                local data=$1;
                local data_dir=$( dirname ${data} );

                if [ "${data_dir}" == "." ]; then
                        data=${PWD}"/"$( basename ${data} );
                elif [ "${data_dir:0:2}" == ".."  ]; then
                        data=$( dirname ${PWD})"${data:2:${#data}}"
                elif [ "${data_dir:0:1}" == "."  ]; then
                        data=${PWD}"${data:1:${#data}}"
                else
                        case $data in   /*) printf "" ;;   *) data=${PWD}"/"${data} ;; esac
                fi

                echo ${data};
                };

read_jsonfield () {
if [ $# -lt 2 ]; then
echo $0: usage: "read_jsonfield <field> <filename.json> "
return 1;
fi

local field=${1}
local filename=${2}

local varfile_orig=$( cat ${filename} )
local idx0=$( str_index "${varfile_orig}" "${field}" )
varfile_0=${varfile_orig:${idx0}}
local idx1=$( str_index "${varfile_0}" ":" )
local idx2=$( str_index "${varfile_0}" "," )
local varfile=${varfile_0:${idx1}+1:${idx2}-${idx1}-1}
local idx_=$( str_index "${varfile}" "[" )
if [ ${idx_} -ne -1 ]; then
local idx1=$( str_index "${varfile_0}" "[" )
local idx2=$( str_index "${varfile_0}" "]" )
local varfile=${varfile_0:${idx1}+1:${idx2}-${idx1}-1}
fi
varfile=$( echo ${varfile//' '/''} )
echo ${varfile//' '/''}
}
