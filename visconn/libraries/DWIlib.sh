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

track_getMask() {
    if [ $# -lt 2 ]; then
        echo "Usage: track_getMask <track_in.trk> <mask_out.nii.gz> [reference.nii.gz]"
        return 1
    fi

    local input1="$1"
    local input2="$2"
    local input3="${3:-}"

    python << END
import sys
import os
import nibabel as nib
import numpy as np

from dipy.tracking.vox2track import streamline_mapping
from dipy.io.streamline import load_tractogram


def loadTrk(track_filename):
    sft = load_tractogram(track_filename, 'same', bbox_valid_check=False)
    streamlines = sft.streamlines
    affine = sft.space_attributes[0]
    header = sft.space_attributes
    return streamlines, affine, header


def track2mask(track_filename, output_filename=None, structural_filename=None):
    track, track_aff, _ = loadTrk(track_filename)

    if structural_filename is None or len(structural_filename) == 0:
        sft = load_tractogram(track_filename, 'same', bbox_valid_check=False)
        sft.to_vox()
        sft.to_corner()
        transformation, dimensions, _, _ = sft.space_attributes

        stream_map = streamline_mapping(track, affine=track_aff)
        Points = np.zeros(dimensions, dtype=np.uint8)

        for idx in stream_map.keys():
            Points[idx[0], idx[1], idx[2]] = 1

        if output_filename is not None:
            nib.save(nib.Nifti1Image(Points, transformation), output_filename)

    else:
        struct_nib = nib.load(structural_filename)
        affine = struct_nib.affine
        header = struct_nib.header
        struct_data = struct_nib.get_fdata()

        stream_map = streamline_mapping(track, affine=affine)
        Points = np.zeros(struct_data.shape, dtype=np.uint8)

        for idx in stream_map.keys():
            Points[idx[0], idx[1], idx[2]] = 1

        if output_filename is not None:
            nii_output = nib.Nifti1Image(Points, affine=affine, header=header)
            nii_output.to_filename(output_filename)

    return Points


track2mask("$input1", "$input2", "$input3")
END
}


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
