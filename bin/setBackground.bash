#!/bin/bash
#
# Usage: setBackground [--killall] [--limitcpu PERCENT] [ any options to pass through ]

# Run the makeSpanningBackground program with any user-selected options.  This
# is an easy-to-use frontend for the full makeSpanningBackground program on
# Linux systems.  Set the paths in the user-settable variables section and copy
# the script to your bin directory as 'setBackground'.  Be sure to make it
# executable with 'chmod +x ~/bin/setBackground'.
#
# Note that two optional arguments are added to the normal
# makeSpanningBackground options.  They must be the first arguments to the
# command if they are used.  The first, '--killall', kills any already-running
# makeSpanningBackground processes.  The second, '--limitcpu', takes one
# argument, which should be an integer percentage value.  It limits the CPU
# usage to that percentage, provided that the cpulimit program is installed.
# Any arguments after these two optional ones are passed unchanged to the
# makeSpanningBackground program.
#
# On Ubuntu I set this script as a startup program to randomly change the
# backgrounds every three hours, using these settings:
#
#    /home/myusername/bin/setBackground --killall --limitcpu 20 -t 180
#
# The script can still be called at any other time, as setBackground, for
# one-time uses such as when you are tired of the current background images.

##############################################
###### User-settable variables ###############
##############################################

# SET THESE VARIABLES FOR YOUR SYSTEM

pathToMakeSpanningBackground="~/SET_THIS_PATH_FOR_YOUR_SYSTEM/makeSpanningBackground.py"
backgroundFilesAndDirs="~/Pictures" # a colon-separated list of files and dirs
outputFilePath="~/Pictures/displayComboImage.bmp" # where to write the combined image
extraOptionsToAlwaysUse="" # any extra option arguments you want to always use

imageQuality=5 # 0 is fastest but lowest quality, 5 is the highest quality
nicenessLevel=10 # 0 is ordinary priority, 19 is nicest
pathForLogcurrentFile="~/Pictures/currentImagesLog.txt" # where to write log file
nameOfProgramInProcessList="makeSpanningBackground.py" # the name so kill can find it

##############################################
###### No user-settable variables below ######
##############################################

# Process the command-line arguments.
while true
do
   if [ "$1" == "--killall" ]; then
      killall -q $nameOfProgramInProcessList; shift
   elif [ "$1" == "--limitcpu" ]; then
      limitPercent=$2; shift; shift
   else break
   fi
done

# Directory names and filenames might have spaces in them.  This converts the
# colon-separated list backgroundFilesAndDirs into a list of quoted pathnames.
IFS=':' read -a dirnameArray <<< "$backgroundFilesAndDirs"
quotedBackgroundFileAndDirList=""
for name in "${dirnameArray[@]}"
do
   quotedBackgroundFileAndDirList="$quotedBackgroundFileAndDirList '$name'"
done

# Expand any tilde usernames in the path to the makeSpanningBackground program.
# (The makeSpanningBackground program can handle any others internally.)
eval progName="$pathToMakeSpanningBackground"

# Run the command.  Note that how it is run will affect how it appears in the
# process list, which is important for the --killall option.
nice -$nicenessLevel $progName \
       $extraOptionsToAlwaysUse \
       -L "$pathForLogcurrentFile" \
       -o "$outputFilePath" \
       -c 0 0 0 -z $imageQuality \
       $quotedBackgroundFileAndDirList "$@" &

#renice -$nicenessLevel -p $! 

if [ "$limitPercent" != "" ]; then
   # Limit CPU usage.  This slows things but can help to maintain
   # responsiveness and keep the fan quiet.
   cpulimit -b -z -p $! --limit=$limitPercent 2>&1 >/dev/null
fi

