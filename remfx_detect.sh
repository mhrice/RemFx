#! /bin/bash

# Example usage:
# ./remfx_detect.sh wet.wav -o examples/output.wav
# first argument is required, second argument is optional

# Check if first argument is empty
if [ -z "$1" ]
then
  echo "No audio input path supplied"
  exit 1
fi

audio_input=$1
# Shift first argument away
shift
output_path=""

while getopts ":o:" opt; do
  case $opt in
    o)
      output_path=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done


# Run script
# If output path is blank, leave it blank

if [ -z "$output_path" ]
then
  python scripts/remfx_detect.py +exp=remfx_detect +audio_input=$audio_input
  exit 0
fi
python scripts/remfx_detect.py +exp=remfx_detect +audio_input=$audio_input +output_path=$output_path
