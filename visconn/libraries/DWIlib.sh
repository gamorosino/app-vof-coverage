#! /bin/bash

#########################################################################################################################
#########################################################################################################################
###################                  ###################
###################title:              DWI library              ###################
###################                 ###################
###################description:Library of functions for DWI / tractography processing        ###################
###################      ###################
###################version:0.2.0                              ###################
###################bash version:   tested on GNU bash, version 4.2.53      ###################
###################autor: gamorosino           ###################
###################     email: g.amorosino@gmail.com      ###################
#########################################################################################################################
#########################################################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"
IMAGINGlib=${SCRIPT_DIR}"/IMAGINGlib.sh"
STRlib=${SCRIPT_DIR}"/STRlib.sh"
FILElib=${SCRIPT_DIR}"/FILElib.sh"
ARRAYlib=${SCRIPT_DIR}"/ARRAYlib.sh"

source "${IMAGINGlib}"
source "${STRlib}"
source "${FILElib}"
source "${ARRAYlib}"


track_getMask () {
                #############################################################
                # track_getMask — Generate a binary WM mask from a tractogram
                #
                # Usage: track_getMask <tractogram.tck> <out_mask.nii.gz> <reference.nii.gz>
                #
                # Steps:
                #   1. tckmap  → track-density image (TDI)
                #   2. fslmaths -bin  → binary mask
                #############################################################

if [ $# -lt 3 ]; then
    echo $0: "usage: track_getMask <tractogram.tck> <out_mask.nii.gz> <reference.nii.gz>"
    return 1;
fi

local tractogram="${1}"
local out_mask="${2}"
local reference="${3}"

local out_dir=$( dirname "${out_mask}" )
mkdir -p "${out_dir}"

local stem=$( basename "${out_mask}" .nii.gz )
local tdi_tmp="${out_dir}/${stem}_tdi_tmp.nii.gz"

echo "[track_getMask] tractogram : ${tractogram}"
echo "[track_getMask] reference  : ${reference}"
echo "[track_getMask] output     : ${out_mask}"

# Step 1: create track-density image
tckmap "${tractogram}" "${tdi_tmp}" \
-template "${reference}" \
-quiet -force

# Step 2: binarise
fslmaths "${tdi_tmp}" -bin "${out_mask}"

rm -f "${tdi_tmp}"

echo "[track_getMask] Done: ${out_mask}"
};


track_filter_length () {
                #############################################################
                # track_filter_length — Keep streamlines within [min, max] mm
                #
                # Usage: track_filter_length <in.tck> <out.tck> <min_mm> [<max_mm>]
                #############################################################

if [ $# -lt 3 ]; then
    echo $0: "usage: track_filter_length <in.tck> <out.tck> <min_mm> [<max_mm>]"
    return 1;
fi

local in_tck="${1}"
local out_tck="${2}"
local min_len="${3}"
local max_len="${4:-10000}"

tckedit "${in_tck}" "${out_tck}" \
-minlength "${min_len}" \
-maxlength "${max_len}" \
-quiet -force

echo "[track_filter_length] ${in_tck} → ${out_tck}  [${min_len}, ${max_len}] mm"
};


track_count () {
                #############################################################
                # track_count — Print the number of streamlines in a .tck file
                #
                # Usage: track_count <tractogram.tck>
                #############################################################

if [ $# -lt 1 ]; then
    echo $0: "usage: track_count <tractogram.tck>"
    return 1;
fi

local tck="${1}"
tckinfo "${tck}" -count | tail -1 | awk '{print $NF}'
};
