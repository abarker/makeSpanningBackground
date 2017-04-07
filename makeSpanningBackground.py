#!/usr/bin/python
# Note that at least on Ubuntu 12.04 the shebang "/usr/bin/env python" does not
# properly set the Linux process name to "makeSpanningBackground" for utilities
# such as ps, top, and killall (it is set to "python", which is difficult to
# distinguish from other running apps).
"""

makeSpanningBackground -- set the background wallpaper for multiple monitors

A command-line application which sets the background wallpaper on multiple
monitors.   By default the selected wallpapers will be automatically scaled,
combined into a single, large image, and displayed as the background wallpaper
on the active monitors.  Many options are available, including the option to
set a time and regularly change the wallpapers.

Run
   makeSpanningBackround.py -h | more
or
   python makeSpanningBackground.py -h | more
to see the formatted documentation.

An example usage:
   makeSpanningBackground.py ~/backgroundDir -o outputImage.bmp

Copyright (c) 2012 Allen Barker, released under the MIT license.
Project web site: http://abarker.github.com/makeSpanningBackground
Source code site: https://github.com/abarker/makeSpanningBackground
See http://opensource.org/licenses/MIT or the file LICENSE in the
source directory for the text of the license.

"""

# This program uses the Scipy image class ndimage for image processing.  It
# could have used PIL alone (scipy.ndimage depends on PIL), but it started as
# an exercise in using numpy.  For detailed docs on ndimage, see
# http://docs.scipy.org/doc/scipy/reference/tutorial/ndimage.html
#
# Note that the ndimages read by sp.ndimage.imread actually *are* Numpy ndarray
# objects.  They are just interpreted as images and operated-on by the ndimage
# functions.  An RGB image can be treated as a 2-D array of three-element 1-D
# arrays (where the three elements store RGB value as unsigned bytes).  Or, it
# can be treated as a 3-D array of unsigned bytes (with the last dimension
# indexing three "planes" for the color space).
#
# It is important to note that indexing on ndimages reverses the usual
# img[x][y] convention, to img[y][x].  The ndimage "shape" function
# correspondingly returns the shape with this reversed convention.  That is,
# the format returned by shape is (maxY, maxX, 3), where 3 is the depth of RGB
# byte values.  In order to reduce confusion this current program also adopts
# the ndimage convention.
#
#
# Future enhancements, maybe:
# 1) Add the option to base scalings on physical sizes (available from xrandr).
# 2) Have Windows switch to tiled mode automatically (not working in Win 7).
# 3) Use threading on the scaling calculations, for multi-core speedup.
# 4) Easy way to mod the interface for separate dir/files for each display.
# 5) Allow URLs to image files as args (PIL can handle urllib-opened links).

from __future__ import division, print_function
try:
    import numpy as np
    import scipy as sp
except ImportError:
    print("Error: pdfCropMargins requires that the SciPy and NumPy packages"
          "\nboth be installed in Python.  There was an error on the import,"
          "\nwhich suggests that at least one is missing.\n",
          file=sys.stderr)
    raise
import scipy.ndimage # pyflakes gives error, but this is needed
import subprocess
import os
import os.path
import sys
import random
import time
try:
    from PIL import Image
except ImportError:
    print("Could not import the Python Imaging Library (PIL)."
          "\nPIL must be installed to use this program.\n",
          file=sys.stderr)
    raise
# import matplotlib.pyplot as plt # only needed for debugging, to view images

#
# Get info about the OS we're running on.
#

import platform
python_version = platform.python_version_tuple()
system_os = platform.system() # "Linux" or "Windows"

if system_os == "Windows":
    import ctypes # The ctypes package is included in Python 2.5 and higher.
    windows_version = platform.win32_ver()
    windows_release = platform.release()
    yx_primary_window_origin = () # global tuple, set by get_display_info_windows()
elif system_os == "Linux":
    linux_version = platform.linux_distribution()

#
# Set up some basic image-processing stuff.
#

# These are the allowed image suffixes, which the PIL library (used by ndimage)
# can read (these are the common ones, but it can actually read more).
# The same suffixes with all uppercase are also accepted, but mixed-case is not.
allowed_image_file_suffixes = [
    ".bmp", ".dib", ".dcx", ".eps", ".ps", ".gif", ".im", ".jpg", ".jpe",
    ".jpeg", ".pcd", ".pcx", ".pdf", ".png", ".pbm", ".pgm", ".ppm", ".psd",
    ".tif", ".tiff", ".xbm", ".xpm"]

allowed_image_file_suffixes += [s.upper() for s in allowed_image_file_suffixes]
allowed_image_file_suffixes = set(allowed_image_file_suffixes)

def name_has_image_suffix(fname):
    """Test whether file fname has an image suffix in the allowed list."""
    extension = os.path.splitext(fname)[1]
    return extension in allowed_image_file_suffixes


#
# Some general utility functions.
#


def process_path(path):
    """This removes any outer quotes, does an expanduser, and returns the
    absolute path."""
    if path.startswith('"') and path.endswith('"'): path = path[1:-1]
    if path.startswith("'") and path.endswith("'"): path = path[1:-1]
    return os.path.abspath(os.path.expanduser(path))

def path_error_exit(path, msg):
    """Exit the program due to a path error, printing a message."""
    print("\nError in makeSpanningBackground related to this pathname:\n   "
          + path + "\n" + msg, file=sys.stderr)
    sys.exit(1)


################################################################################
##
# Begin command-line parsing routines and help-message documentation.
##
# Everything from here down to the matching end-comment could be a separate
# module but it is included directly so that the program fits into a
# single file (as a simple-to-run script).
##
# General argparse reminders and warnings:
# 1) Do not use unescaped percent signs in the documentation strings.
# 2) Using nargs=1 causes the values to be placed in a list, default doesn't.
# 3) First argument specified is the one which appears in the Usage message.
# 4) The metavar kwarg sets the string for option's VALUES in Usage messages.
# 5) With a default value set you can always assume some value is assigned.
# 6) Use numargs=1 and default=[] to test on the var for whether or not, say,
# an int-valued option was selected at all (or check for value None).
##
################################################################################

import argparse
import textwrap
import re

#
# Define classes to allow redirecting sys.stdout and sys.stderr in order to
# postprocess (prettify) the help and usage messages from the argparse class.
# Also define a self-flushing output stream to avoid having to explicitly run
# Python with the '-u' option in Cygwin terminals.
#
# Note that the standard TextWrapper fill and wrap routines used in argparse
# do not strip out multiple whitespace like many fill programs do.
#


class RedirectHelp(object):
    """This class redirects stdout in order to prettify the output
    of argparse's help and usage messages (via a postprocessor).  The
    postprocessor does a string replacement for all the pairs defined in the
    user-defined sequence help_string_replacement_pairs.  It also adds the following
    directives to the formatting language:
       ^^s          replaced with a space, correctly preserved by ^^f format
       \a           the bell control char is also replaced with preserved space
       ^^f ... ^^f  reformat all the text between these directives
       ^^n          replaced with a newline (after any ^^f format)
    Formatting with ^^f converts any sequence of two or more newlines into a
    single newline, i.e., a paragraph break.  Multiple (non-preserved)
    whitespaces are converted to a single white space, and the text in
    paragraphs is line-wrapped with a new indent level."""

    def __init__(self, outstream, help_string_replacement_pairs=(),
                 init_indent=5, subs_indent=5, line_width=76):
        """Will usually be passed sys.stdout or sys.stderr as an outstream
        argument.  The pairs in help_string_replacement_pairs are all applied to the
        any returned text as postprocessor string replacements.  The initial
        indent of formatted sections is set to init_indent, and subsequent indents
        are set to subs_indent.  The line width in formatted sections is set to
        line_width."""
        self.outstream = outstream
        self.help_string_replacement_pairs = help_string_replacement_pairs
        self.init_indent = init_indent
        self.subs_indent = subs_indent
        self.line_width = line_width

    def write(self, s):
        pretty_str = s
        for pair in self.help_string_replacement_pairs:
            pretty_str = pretty_str.replace(pair[0], pair[1])
        # Define ^^s as the bell control char for now, so fill will treat it right.
        pretty_str = pretty_str.replace("^^s", "\a")

        def do_fill(match_obj):
            """Fill function for regexp to apply to ^^f matches."""
            st = pretty_str[match_obj.start()+3:match_obj.end()-3] # get substring
            st = re.sub("\n\s*\n", "^^p", st).split("^^p") # multi-new to para
            st = [" ".join(s.split()) for s in st] # multi-whites to single
            wrapper = textwrap.TextWrapper( # indent formatted paras
                initial_indent=" "*self.init_indent,
                subsequent_indent=" "*self.subs_indent,
                width=self.line_width)
            return "\n\n".join([wrapper.fill(s) for s in st]) # wrap each para
        # do the fill on all the fill sections
        pretty_str = re.sub(r"\^\^f.*?\^\^f", do_fill, pretty_str, flags=re.DOTALL)
        pretty_str = pretty_str.replace("\a", " ") # bell character back to space
        pretty_str = pretty_str.replace("^^n", "\n") # replace ^^n with newline
        self.outstream.write(pretty_str)
        self.outstream.flush() # automatically flush each write

    def __getattr__(self, attr):
        return getattr(self.outstream, attr)


class SelfFlushingOutstream(object):
    """This class allows stdout and stderr to be redefined so that they are
    automatically flushed after each write.  (The same thing can be achieved via
    the '-u' flag on the Python command line.)  This helps when running in
    Cygwin terminals.  Class is independent of the RedirectHelp class above."""

    def __init__(self, outstream):
        """Will usually be passed sys.stdout or sys.stderr as an argument."""
        self.outstream = outstream

    def write(self, s):
        self.outstream.write(s)
        self.outstream.flush()

    def __getattr__(self, attr):
        return getattr(self.outstream, attr)


def parse_command_line_arguments(argparse_parser, help_string_replacement_pairs=(),
                              init_indent=5, subs_indent=5, line_width=76):
    """Main routine to call to execute the command-line parsing.  Returns an
    object from argparse's parse_args() routine."""
    # Redirect stdout and stderr to prettify help or usage output from argparse.
    old_stdout = sys.stdout # save stdout
    old_stderr = sys.stderr # save stderr
    sys.stdout = RedirectHelp(
        sys.stdout, help_string_replacement_pairs, init_indent, subs_indent,
        line_width) # redirect stdout to add postprocessor
    sys.stderr = RedirectHelp(
        sys.stderr, help_string_replacement_pairs, init_indent, subs_indent,
        line_width) # redirect stderr to add postprocessor

    # Run the actual argument-parsing operation via argparse.
    args = argparse_parser.parse_args()

    # The argparse class has finished its argument-processing, so now no more
    # usage or help messages will be printed.  So restore stdout and stderr
    # to their usual settings, except with self-flushing added.
    sys.stdout = SelfFlushingOutstream(old_stdout)
    sys.stderr = SelfFlushingOutstream(old_stderr)

    return args

###############################################################################
##
# End of command-line parsing routines and documentation.
##
###############################################################################

#
# Define and document the command-line arguments and flags with argparse.
# Note that the default is for raw, i.e., unformatted text unless the
# formatting directive ^^f is specified.
#

# TODO: add the metavar kwarg where it helps

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="""
Description:

^^f
   Make a single, combined background image file from separate background image
   files, one for each display.  The default is to get information from the
   system about the display resolutions, and to randomly choose an image for
   each display from the image files and image directories provided.  Sampling
   is without replacement; the full list is regenerated when it is empty
   (including any changes which might have occurred in the directories).

   The positional arguments are image files and directories containing image
   files.  The required flag '--outfile' or '-o' specifies the pathname of the
   file where the combined image file will be written (or silently
   overwritten).  An example:

   \a\a\apython makeSpanningBackground.py ~/bgDir photo.jpg -o combo.bmp

   On Linux the program file can be made executable and the initial "python"
   above can be omitted if the Python version at "/usr/bin/python" is the
   correct one (if not, the first line in the program can be modified).
   Windows users can achieve the same effect by renaming the script to have a
   '.py' suffix and making sure '.PY' is in the PATHEXT list.  On any system
   the program file (or a script calling it) can be placed somewhere in the
   PATH to avoid having to supply the path.

   To use this kind of combined background image in Linux the background mode
   should be set to 'spanned'; the program will attempt to set this
   automatically.  On Windows the background mode should be 'tiled'.  The
   program will attempt to do this, but if that fails the mode must be
   explicitly set by the user (usually from the system's background-setting
   window).  In Windows XP the '.bmp' format should be chosen for the output
   file (simply by using that suffix on the output filename).  Backgrounds set
   in Windows XP do not persist between logins and must be reset.

   This program can be set as a startup program to change the background
   wallpaper on logins.  When the '-t <minutes>' command switch is used the
   program will infinitely loop, waiting for the specified number of minutes
   between iterations.

   It is convenient to create a simple shell-script wrapper or batch file to
   call the program with the "usual" command-line arguments.  One for Linux
   is packaged with the program in the bin directory.
^^f
""",

    epilog="""The makeSpanningBackground program is Copyright (c) 2012 by Allen Barker.
Released under the permissive MIT license.""")

parser.add_argument("image_files_and_dirs", nargs="+",
                    metavar="IMAGE_FILE_OR_DIR", help="""

   A whitespace-separated list of the pathnames of image files and/or
   directories containing image files.  Use quotes around any file or directory
   name which contains a space.  Pathnames can be repeated on the list, and the
   list will be reloaded if it becomes empty.  A specified filename will be
   silently ignored if it does not have a suffix in the list """ +
                    str(sorted(list(allowed_image_file_suffixes))) + ".^^n")

parser.add_argument("-o", "--outfile", required=True, nargs=1,
                    metavar="OUTFILE_NAME", help="""

   The pathname of the output file.  This flag is required.  Any existing file
   by that name will be silently overwritten.  The filename can have any suffix
   that ndimage (via PIL) can handle, and the output file will be written in
   that format.  Not all formats will necessarily be settable as background
   images.  Common choices are '.jpg' and '.bmp', where the latter produces
   files which are larger but of higher quality.  Note that Windows XP requires
   the '.bmp' format.^^n""")

parser.add_argument("-v", "--verbose", action="store_true", help="""

   Print more information about the program's actions and progress.  Without
   this switch only error messages are printed to the screen.^^n""")

parser.add_argument("-1", "--oneimage", action="store_true", help="""

   Use a single background image, scaled to stretch over all the displays.
   This is currently based on the screen resolutions, not the physical sizes of
   the monitors (dpi).^^n""")

parser.add_argument("-f", "--fitimage", nargs=3, type=int,
                    metavar=("R_VAL", "G_VAL", "B_VAL"), help="""

   If this option is set then images are scaled to fully fit into the
   corresponding display.  The three arguments are RGB byte values specifying
   the color of any region not covered by an image.  Use '-f 0 0 0' for black.
   The default program behavior without this switch is to use the minimum
   scaling to completely fill the display while preserving the aspect
   ratio.^^n""")

parser.add_argument("-t", "--timedelay", nargs=1, type=float, metavar="MINUTES",
                    help="""

   If this option is set the program will infinitely loop.  On each iteration
   it will collect the current display information, re-select images for each
   display, create a combined image, and set it as the background (assuming
   that behavior is not modified by any other options).  The floating point
   argument gives the number of minutes to sleep between iterations.  The image
   list will only be reloaded when it becomes empty, so every image will be
   used exactly once before any images are reloaded.^^n""")

parser.add_argument("-p", "--percenterror", nargs=1, type=float,
                    metavar="PCT_FLOAT", help="""

   The percentage of an image's area which is allowed to be cropped-out when
   scaling it to fit a display resolution.  In '--fitimage' mode it is the
   percent of non-image in the display area.  In '--oneimage' mode it is the
   percent of the image that does not fall onto any display.  Set this option
   to zero for exact fit only.  The program will exit with an error message if
   it cannot find enough suitable images.^^n""")

parser.add_argument("-z", "--zoomspline", nargs=1, type=int,
                    metavar="ONE_OF_012345", help="""

   Set the order of the spline used by ndimage to resize (zoom) images.  The
   value can be from 0 to 5, with 3 the default.  Lower orders are faster,
   higher orders have better quality.  (Also, for higher-quality images: output
   files in '.bmp' format are larger but tend to look better.)^^n""")

parser.add_argument("-s", "--sequential", action="store_true", help="""

   Process images sequentially as listed in the positional arguments, with the
   files in any image directories forming alphabetical sublists.  This can be
   used to force specific images to appear on specific displays.  On Linux the
   ordering of the displays is the same as in xrandr.  If you are unsure about
   the ordering, try some images and see where they end up.  If display
   resolutions and offsets are explicitly set with '--reslist' and
   '--sequential' is set then the image files will corresponding one to one
   with the list of resolution specifiers.^^n""")

parser.add_argument("-c", "--colorfill", nargs=3, type=int,
                    metavar=("R_VAL", "G_VAL", "B_VAL"), help="""

   The three arguments are RGB byte values specifying the color of any region
   in the large, combined image (a bounding-box on the displays) which is not
   covered by a display.  Use '-c 0 0 0' for black.^^n""")

parser.add_argument("-R", "--recursive", action="store_true", help="""

   Recursively search any supplied image directories for image files.^^n""")

parser.add_argument("-d", "--dontapply", action="store_true", help="""

   Do not attempt to apply the created image as the working background image.
   The program will simply exit after writing the image file.^^n""")

parser.add_argument("--noclobber", action="store_true", help="""

   Never overwrite an existing file as the output file.^^n""")

parser.add_argument("-r", "--reslist", nargs="+", metavar="RESLIST", help="""

   Set the resolutions and offsets to use. No system lookup will be attempted.
   This must be a space-separated list of strings of the form
   x*y+xOffset+yOffset (like in the output of xrandr on Linux), where xOffset
   and yOffset give the top-left position of the display and x and y are the
   display sizes (resolution).  Like in X11, (0,0) is assumed to be at the top
   left of a bounding box on all the displays.  For a Windows machine the top
   left of the primary display will also need to be defined, using the
   '--windows' option described below.  This option cannot be immediately
   followed by the positional arguments.^^n""")

parser.add_argument("-w", "--windows", nargs=2, type=int,
                    metavar=("X_POS", "Y_POS"), help="""

   This option should only be necessary on a Windows machine when using the
   '--reslist' option. The two arguments x and y (in that order) should be set
   to the top left position of the primary display.  This is necessary because,
   unlike X11, Windows assumes (0,0) is at the top left of the primary display.
   Other displays can then have negative locations.  This program translates
   Windows positions so that all addresses are positive, with (0,0) at the top
   left of all the displays (which is how positions should be entered for the
   '--reslist' option).  The program then corrects for this translation in a
   final step, which wraps the image into tiled mode relative to the primary
   display.  This requires knowing the top left position of the primary
   display.^^n""")

parser.add_argument("-x", "--x11", action="store_true", help="""

   Create an X11 type image even if Windows is detected as the OS.  When this
   option is selected no final "modular wrap" will be applied to correct for
   the position of the primary display.^^n""")

parser.add_argument("-L", "--logcurrent", nargs=1, metavar="FNAME", help="""

   Write the names of the current images to a file.  The single argument is the
   name of the file to write the filenames to.  Useful when you want to know
   the filenames of the images being displayed.""")

#
# Define some prettifying modifications to the usual help output of argparse.
#

prog_name = "makeSpanningBackground.py" # Separate out string, in case it changes.
help_string_replacement_pairs = (
    ("usage: ", "^^nUsage: "),
    ("positional arguments:", "Positional arguments:^^n"),
    ("optional arguments:", "Optional arguments:^^n"),
    ("show this help message and exit",
     "Show this help message and exit.^^n"),
    ("%s: error: too few arguments" % prog_name,
     textwrap.fill("^^nError in arguments to %s: "
                   "image source and output file arguments are required.^^n"
                   % prog_name)),
    (prog_name + ": error:", "Error in "+prog_name+":")
)


#
# Functions which do the real work.
#

def get_display_info():
    """OS-independent function call for getting display information."""

    # If the user specified a resolution list, parse it and return those values.
    if args.reslist:
        display_res_list = [a.replace("+", "x").split("x") for a in args.reslist]
        # convert all to ints
        display_res_list = [[int(b) for b in a] for a in display_res_list]
        # swap x and y ordering
        display_res_list = [[a[1], a[0], a[3], a[2]] for a in display_res_list]
        return tuple(tuple(i) for i in display_res_list) # convert to tuples

    if system_os == "Linux":
        return get_display_info_linux()
    elif system_os == "Windows":
        return get_display_info_windows()
    else:
        print("\nWarning from makeSpanningBackground: System OS not recognized."
              "\nMaybe try setting resolution explicitly with the '--reslist'"
              "\noption.  Assuming an 'xrandr' system and hoping for the best.",
              file=sys.stderr)
        return get_display_info_linux()


def get_display_info_linux():
    """Parse the output of xrandr to get the positions and resolutions of the
    displays.  Returns a tuple of 4-tuples, one for each active display.  Each
    4-tuple is in the form (y, x, yOffset, xOffset) where the values are taken
    from the string of the form x*y+xOffset+yOffset.  The unconventional
    ordering is for conformity with the convention of the ndimage shape
    command."""

    # The multiple screens are essentially treated like one giant screen, with
    # the "origin" for each screen and the screens' resolutions determining the
    # layout within the giant screen.  Screens can essentially be arbitrarily
    # arranged on the giant-screen canvas (like in display-settings programs).
    # The origin of the giant screen is at the top, leftmost point.  We only
    # need to consider a bounding box containing all the smaller screens.
    #
    # Left-to-right ordering, for example, is determined by the additive x
    # offsets in the resolution for each display, in a format like
    # x*y+xOffset+yOffset.
    #
    # Assume left, inverted for all displays (though that could be calculated
    # from xrandr output without too much additional trouble).
    #
    # Note that we need to look for asterisks in the xrandr output, because when
    # a monitor is connected but not selected in the display settings xrandr
    # will report it as connected, but an asterisk will not appear anywhere in
    # that section.

    try:
        output = subprocess.check_output("xrandr").decode("utf-8")
    except OSError as e:
        print("\nError running the 'xrandr' program.  Be sure it is installed."
              "\nIf failures continue, try the '--reslist' option to explicitly"
              "\nset the display resolutions.  The system reports:\n",
              e, file=sys.stderr)
        sys.exit(1)
    output = output.splitlines() # split into output into lines
    output = [line.split() for line in output] # split lines into words

    display_res_list = [] # list of resolution tuples for each display
    for line in output:
        if len(line) < 2:
            continue # just in case format changes
        if line[1] == "connected":
            try:
                while not line[2][0].isdigit():
                    line = line[1:]
            except IndexError:
                continue
            res = line[2] # get string for display's resolution and offsets
            res = res.replace("+", "x").split("x") # split, say, 1024x768+1024+0
            try: # see if res contains int resolution values
                res = [int(val) for val in res] # convert all to ints
            except ValueError: # if not, assume display is not active and skip
                continue
            res[0], res[1] = res[1], res[0] # swap to match ndimage convention
            res[2], res[3] = res[3], res[2] # swap to match ndimage convention
            continue
        # only accept display with resolution res if an asterisk found after it
        asterisk_in_line = [word for word in line if "*" in word]
        if asterisk_in_line:
            display_res_list.append(res)

    return tuple(tuple(i) for i in display_res_list) # convert lists to tuples


def get_display_info_windows():
    """Get the display resolutions and offsets on a Windows system.  This is
    based on http://code.activestate.com/recipes/460509/, with modifications.
    Note that several classes and functions are defined inside this function
    since they are not needed anywhere else (and this function is only run once
    per program execution)."""

    # import ctypes # imported at the top of the file
    # windll seems to change to pydll in Cygwin python, for future
    user = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = [
            ('left', ctypes.c_long),
            ('top', ctypes.c_long),
            ('right', ctypes.c_long),
            ('bottom', ctypes.c_long)]

        def dump(self):
            # note the left, top, right, bottom ordering in the dump
            return [int(i) for i in (self.left, self.top, self.right, self.bottom)]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [
            ('cbSize', ctypes.c_ulong),
            ('rcMonitor', RECT),
            ('rcWork', RECT),
            ('dwFlags', ctypes.c_ulong)]

    def get_monitors():
        retval = []
        # WINFUNCTYPE seems change to CFUNCTYPE in Cygwin Python, for future
        CBFUNC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                                    ctypes.POINTER(RECT), ctypes.c_double)

        def cb(h_monitor, hdc_monitor, lprc_monitor, dw_data):
            r = lprc_monitor.contents
            data = [h_monitor]
            data.append(r.dump())
            retval.append(data)
            return 1
        cbfunc = CBFUNC(cb)
        temp = user.EnumDisplayMonitors(0, 0, cbfunc, 0) # unused var, side effect?
        return retval

    def monitor_areas():
        retval = []
        monitors = get_monitors()
        for h_monitor, extents in monitors:
            data = [h_monitor]
            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            mi.rcMonitor = RECT() # the full monitor size
            mi.rcWork = RECT()    # the working area (minus toolbars, etc.)
            res = user.GetMonitorInfoA(h_monitor, ctypes.byref(mi)) # unused var, side effect?
            data.append(mi.rcMonitor.dump())
            data.append(mi.rcWork.dump())
            retval.append(data)
        return retval

    all_areas = monitor_areas()
    use_working_area = False # use the full monitor area
    if use_working_area:
        index = 2
    else:
        index = 1

    # The primary window on Windows has top left at (0,0), and others can be
    # negative.  Translate to make all coords positive, with (0,0) the top left.
    # But set the global variable yx_primary_window_origin to the original origin's
    # translated location, so we can compensate later.  Also, convert to X11
    # size and offset format (with swapped x,y for ndimage compatibility).

    min_x = min([min(a[index][0], a[index][2]) for a in all_areas])
    min_y = min([min(a[index][1], a[index][3]) for a in all_areas])
    global yx_primary_window_origin
    yx_primary_window_origin = (-min_y, -min_x)
    display_res_list = []
    for area in all_areas:
        a = area[index]
        a = [a[0]-min_x, a[1]-min_y, a[2]-min_x, a[3]-min_y]
        display_res_list.append((a[3]-a[1], a[2]-a[0], a[1], a[0])) # convert

    return tuple(display_res_list)


def reload_background_files():
    """Return the list of paths corresponding to args.image_files_and_dirs
    relative to the current state of the filesystem."""
    all_background_files = []
    for image_path in args.image_files_and_dirs:
        image_path = process_path(image_path) # tilde expand the path

        # Make sure we didn't get passed a bad pathname.
        if not os.path.exists(image_path):
            path_error_exit(image_path, "Path does not exist.")

        # Handle directories of images.
        if os.path.isdir(image_path):
            for dirpath, dirnames, filenames in \
                    os.walk(image_path, followlinks=True):
                dirnames.sort() # alphabetical recurse
                filenames.sort() # alphabetical file ordering
                files_in_dir = [os.path.join(dirpath, f) for f in filenames]
                all_background_files += files_in_dir
                if not args.recursive:
                    break

        # Handle individual image files (note symlinks are treated as files).
        elif os.path.isfile(image_path):
            all_background_files.append(image_path)

        # Ignore if neither file nor directory unless has image suffix.
        else:
            if name_has_image_suffix(image_path):
                path_error_exit(image_path, "Not a file or a directory.")

    # Remove non-image files from list and expand to full pathnames.
    all_background_files = [process_path(f) for f in all_background_files
                            if name_has_image_suffix(f)]
    return all_background_files


all_background_files = [] # global list of all the background files specified


def get_next_background_image(disp_res, disp_res_list):
    """Get the next background image from the user-specified list.  The selected
    file is removed from the global list once it is selected, but only the
    selected instance is removed (files can be listed multiple times as
    arguments).  Returns a 2-tuple containing the filename of the file and the
    image itself.  Always reloads the global list of filenames when the list
    becomes empty.  Returns the empty tuple only when the global list has no
    suitable filenames (such as when '--percenterror' is too low for the
    available images).  The display resolution argument disp_res is used to
    calculate the percent error in scaling when the '--percenterror' option is
    selected, and the list disp_res_list is used to calculate the error when the
    '--oneimage' option is selected."""
    # Create a local list of indices to choose from, so we can remove bad
    # candidate-images locally before finally modifying the global list.
    global all_background_files # global list of background filenames
    background_indices = list(range(len(all_background_files)))
    local_reset_done = False

    # Select a file from the list of files and try to open it; repeat if needed.
    while True:
        if not background_indices:
            if local_reset_done: # return empty list, no files are suitable
                return ()
            else: # reload the file list to search for a suitable image file
                local_reset_done = True
                all_background_files = reload_background_files()
                background_indices = list(range(len(all_background_files)))
                if args.verbose:
                    print("Loading or reloading the list of image files."
                          "\nFound", len(all_background_files), "image filenames.\n")
        if args.sequential:
            file_index = background_indices.pop(0)
        else:
            file_index = background_indices.pop(
                random.randint(0, len(background_indices) - 1))
        selected_filename = all_background_files[file_index]
        try:
            # bg_image = sp.ndimage.imread(selected_filename) # works
            # bg_image = sp.misc.imread(selected_filename) # works, too
            #
            # The below code is essentially taken directly from the definition
            # of sp.ndimage.imread, except that we check the mode and convert
            # to RGB if the mode is something else.  (Copying images in
            # different modes to a common image file can cause problems.)
            fp = open(selected_filename, "rb")
            im = Image.open(fp)
            if im.mode != "RGB":
                im = im.convert("RGB")
            bg_image = np.array(im)
            fp.close()
            # if args.verbose:
            #   print("debug image type, shape, dtype, mode: ",
            #         type(bg_image), bg_image.shape, bg_image.dtype, bg_image.mode)
        except IOError:
            print("\nWarning from makeSpanningBackground: The file\n   " +
                  selected_filename + "\ncannot be read as an image.  Ignoring it.",
                  file=sys.stderr)
            continue
        # Now check for good enough fit (if that option was selected).
        if args.percenterror:
            err_fraction = calculate_scaling(bg_image, disp_res, disp_res_list)[3]
            if err_fraction > args.percenterror[0] / 100:
                if args.verbose:
                    print("Error percentage is " + str(round(err_fraction*100, 1))
                          + "; rejecting image\n   ", selected_filename)
                continue
            else:
                if args.verbose:
                    print("Error percentage is " + str(round(err_fraction*100, 1))
                          + "; accepting image\n   ", selected_filename)
            if args.verbose:
                print()

        # Got a good file, delete it from the full list and exit the loop.
        del all_background_files[file_index]
        break

    return (selected_filename, bg_image)


def copy_subimage(yx_extents, from_image, yx_from_start, to_image, yx_to_start):
    """The arguments from_image and to_image are both ndimages; the others are all
    ordered pairs of y, x positions or sizes.  Copy a subimage of size
    yx_extents from from_image to to_image.  Start the subimage of from_image at
    yx_from_start and start the destination subimage of to_image at yx_to_start.  The
    extents are all "one larger" than the largest index, like range and len.
    The image to_image is directly modified, in-place.  All accesses are compared
    with the shapes of the images, and any out-of-range copy operations are
    silently ignored."""

    # Check bounds; reset any values to ensure all accesses are good.

    # Check from_image bounds.
    yx_extents = list(yx_extents) # may need to reset an extent, tuple to list
    for dim in [0, 1]: # loop over the y=0 and x=1 dimensions
        if yx_from_start[dim] > from_image.shape[dim]:
            yx_extents[dim] = 0
        if yx_from_start[dim] + yx_extents[dim] > from_image.shape[dim]:
            yx_extents[dim] = from_image.shape[dim] - yx_from_start[dim]
    # Check to_image bounds.
    for dim in [0, 1]: # loop over the y=0 and x=1 dimensions
        if yx_to_start[dim] > to_image.shape[dim]:
            yx_extents[dim] = 0
        if yx_to_start[dim] + yx_extents[dim] > to_image.shape[dim]:
            yx_extents[dim] = to_image.shape[dim] - yx_to_start[dim]

    # Do the copy operation.
    to_image[yx_to_start[0]: yx_to_start[0]+yx_extents[0],
            yx_to_start[1]: yx_to_start[1]+yx_extents[1]] = \
        from_image[yx_from_start[0]: yx_from_start[0]+yx_extents[0],
                  yx_from_start[1]: yx_from_start[1]+yx_extents[1]]

    # Nested loop below is equivalent to assignment above, but less efficient.
    # for y in range(yx_extents[0]):
    #    for x in range(yx_extents[1]):
    #       to_image[y+yx_to_start[0]][x+yx_to_start[1]] = \
    #             from_image[y+yx_from_start[0]][x+yx_from_start[1]]
    return


def correct_windows_origin(image):
    """Map an image to conform to Windows conventions (negative positions and
    tiled mode).  Uses the global variable yx_primary_window_origin to access the
    translated origin (to be converted back to (0,0), with proper wraparound for
    any negative pieces)."""

    # imLowY = 0
    im_high_y = image.shape[0]
    # imLowX = 0
    im_high_x = image.shape[1]
    y0 = yx_primary_window_origin[0]
    x0 = yx_primary_window_origin[1]

    # create new giant image
    new_image = np.empty_like(image) # empty, same size as image

    # Note that the translated origin (y0,x0) breaks image into four pieces.
    # We will copy each piece as a chunk, for efficiency.  We could use
    # scipy.ndimage.interpolation.shift, with mode="wrap", except that we
    # don't need interpolation.  Since the routine may not take that into
    # account, this way is probably more efficient.

    # top left piece of image
    new_image[im_high_y-y0:im_high_y, im_high_x-x0:im_high_x] = \
        image[0:y0, 0:x0]
    # bottom right piece of image
    new_image[0:im_high_y-y0, 0:im_high_x-x0] = \
        image[y0:im_high_y, x0:im_high_x]
    # bottom left piece of image
    new_image[0:im_high_y - y0, im_high_x-x0:im_high_x] = \
        image[y0:im_high_y, 0:x0]
    # top right piece of image
    new_image[im_high_y-y0:im_high_y, 0:im_high_x-x0] = \
        image[0:y0, x0:im_high_x]

    return new_image


def scale_image(image, yx_new, zoom_spline):
    """Perform an exact scaling of image, to the int-valued (y,x) sizes in
    yx_new.  Note that the aspect ratio is not considered here; it should be
    (approximately) preserved in calculating yx_new if that is desired."""
    yx_curr = (image.shape[0], image.shape[1])
    if yx_new == yx_curr:
        if args.verbose:
            print("Image is at the correct scale already, skipping the rescale.")
        scaled_image = image
    else:
        # Note zoom seems to truncate down to the nearest image size.  A small
        # additive constant is used to avoid problems due to floating point
        # precision and truncation (e.g., truncating down to 767 instead of
        # producing exactly the selected 768) in ndimage.interpolation.zoom.
        zoom_y = (yx_new[0]+1E-5)/yx_curr[0]
        zoom_x = (yx_new[1]+1E-5)/yx_curr[1]
        if args.verbose:
            print("Scaling image with spline order", zoom_spline, "and "
                  "(zoom_y,zoom_x) =", (round(zoom_y, 4), round(zoom_x, 4)))

        scaled_image = sp.ndimage.interpolation.zoom(
            image, (zoom_y, zoom_x, 1), order=zoom_spline)

        if args.verbose and (scaled_image.shape[0], scaled_image.shape[1]) != yx_new:
            print("Warning: Imperfect scaling in zoom operation.")
    return scaled_image


def calculate_scaling(image, disp_res, disp_res_list):
    """Calculate the scaling of the image to fit a display with resolution
    disp_res.  Returns a 4-tuple containing the current size, the new, scaled
    size (which may be larger than the resolution and need cropping), any
    offsets due to the '--fitimage' option, and the fractional error."""
    yx_curr = (image.shape[0], image.shape[1])
    zoom_y = disp_res[0] / image.shape[0] # zoom to make y fit exactly
    zoom_x = disp_res[1] / image.shape[1] # zoom to make x fit exactly
    yx_exact_y = (disp_res[0], int(round(yx_curr[1]*zoom_y)))
    yx_exact_x = (int(round(yx_curr[0]*zoom_x)), disp_res[1])
    yx_new = yx_exact_y
    if args.fitimage:
        if yx_new[1] > disp_res[1]:
            yx_new = yx_exact_x
            fitimage_offsets = [int(round(abs((yx_new[0]-disp_res[0])/2))), 0]
            err_fraction = (disp_res[0]-yx_new[0])*disp_res[1]/(disp_res[0]*disp_res[1])
        else:
            fitimage_offsets = [0, int(round(abs((yx_new[1]-disp_res[1])/2)))]
            err_fraction = (disp_res[1]-yx_new[1])*disp_res[0]/(disp_res[0]*disp_res[1])
    else:
        if yx_new[1] < disp_res[1]:
            yx_new = yx_exact_x
        fitimage_offsets = [0, 0]
        err_fraction = (yx_new[0]-disp_res[0])*yx_new[1] / (yx_new[0]*yx_new[1]) \
            + (yx_new[1]-disp_res[1])*yx_new[0] / (yx_new[0]*yx_new[1])
    if args.oneimage:
        total_screen_area = sum([d[0]*d[1] for d in disp_res_list])
        err_fraction = abs(yx_new[0]*yx_new[1] - total_screen_area)/total_screen_area

    return (yx_curr, yx_new, fitimage_offsets, err_fraction)


def create_giant_image(image_list, disp_res_list_arg):
    """Create and return the final giant image to set as the combined background
    image.  Each image in image_list is mapped to the corresponding display in
    disp_res_list_arg.  Zoom is done such that each image exactly fits the display on
    at least one dimension; any error in the other dimension is cut off
    equally on both ends."""
    # 1) Find a bounding box of all displays and create a giant image that size.
    # 2) Resize each image to be exactly the size of its corresponding display
    #    (in the one dimension for the selected mode).
    # 3) Copy each resized image to the place in the giant image specified by
    #    its offset information (with extents set to crop any extra from step 2).
    # 4) Convert to Windows tiled mode, if necessary.
    # 5) Return the giant image.

    disp_res_list = disp_res_list_arg[:] # a local copy, modified for oneimage option

    # Find the bounding box around all the displays.
    # minY = 0
    max_y = max([i[0] + i[2] for i in disp_res_list])
    # minX = 0
    max_x = max([i[1] + i[3] for i in disp_res_list])

    # Create the empty giant image.
    if args.verbose:
        print("Creating a large image of size", (max_y, max_x),
              "\nwhich is a bounding box on all the displays.")
    giant_image = np.empty((max_y, max_x, 3), "uint8") # empty display-sized RGB image

    # If oneimage option, reset the display list to represent one large display.
    if args.oneimage:
        disp_res_list = [(max_y, max_x, 0, 0)]

    # Scale all the images to exactly match their corresponding display's
    # resolution (when zoomed/fit according to the selected method).
    scaled_image_list = []
    fitimage_offset_list = [] # extra offsets due to --fitimage, we'll append to it
    for image, disp_res, count in zip(image_list, disp_res_list, range(len(image_list))):
        if args.verbose:
            print()

        # Calculate the scaling.
        yx_curr, yx_new, fitimage_offsets, err = \
            calculate_scaling(image, disp_res, disp_res_list)
        fitimage_offset_list.append(fitimage_offsets)

        # Perform the scaling.
        if args.verbose:
            print("Image", count, "has initial shape", image.shape)
        scaled_image = scale_image(image, yx_new, zoom_spline)

        scaled_image_list.append(scaled_image)

        if args.verbose:
            print("Image", count, "now has shape", scaled_image.shape)

    # Set all pixels to the background fill color, if that option is selected.
    # All the pixels in the initial giant_image are assigned RGB values in a
    # loop; not the most efficient way, but it is simple and it works.
    rgb_fill = []
    if args.colorfill:
        rgb_fill = args.colorfill
    if args.fitimage:
        rgb_fill = args.fitimage
    if rgb_fill:
        rgb_bytes = [np.uint8(i) for i in rgb_fill] # explicitly cast to uint8
        giant_image[:, :] = rgb_bytes # note numpy fill method is for scalar vals
        # giant_image[:][:] = rgb_bytes  # this works, too

    # Copy the central portion of each scaled image to the correct place in the
    # giant image (one dimension may be cut-off automatically by copy routine).
    for scaled_image, disp_res, fitOffsets, count in zip(
                        scaled_image_list, disp_res_list, fitimage_offset_list,
                        range(len(disp_res_list))):
        if not args.fitimage:
            # scaled image won't necessarily all fit, compensate for the overlap
            # (assume scaled size minus disp might be slightly neg, imperfect zoom)
            y_start = (scaled_image.shape[0] - disp_res[0]) / 2
            x_start = (scaled_image.shape[1] - disp_res[1]) / 2
            y_start = int(round(max(0.0, y_start)))
            x_start = int(round(max(0.0, x_start)))
            yx_from_start = (y_start, x_start)
        elif args.fitimage:
            # scaled image will fully fit in the display, scaled above to do so
            yx_from_start = (0, 0)
        yx_extents = (disp_res[0], disp_res[1]) # set extents to display size
        yx_to_start = (disp_res[2] + fitOffsets[0], disp_res[3] + fitOffsets[1])
        if args.verbose:
            print("\nCopying image", count, "from pixel", yx_from_start,
                  "with extents", yx_extents,
                  "\nto the large final image, starting at pixel", yx_to_start)
        # Do the actual copy operation.  Note that if scaled_image is larger than
        # extents (zoomed up) copy_subimage will implicitly crop, and if extents
        # are larger than the size of scaled_image (--fitimage mode) then the
        # extents will be automatically reduced in copy_subimage.
        copy_subimage(yx_extents, scaled_image, yx_from_start, giant_image, yx_to_start)

    if (system_os == "Windows" or args.windows) and not args.x11:
        if args.windows:
            global yx_primary_window_origin
            yx_primary_window_origin = (args.windows[1], args.windows[0])
        if args.verbose:
            print("Correcting the origin of the image (on Windows OS).")
        giant_image = correct_windows_origin(giant_image)

    # plt.imshow(giant_image)
    # plt.show() # for debugging
    return giant_image


def set_image_as_current_wallpaper(image_file_name):
    """Set the file image_file_name to be a spanning background image (in either
    Linux or Windows)."""
    image_file_name = process_path(image_file_name)

    if system_os == "Linux":
        if args.verbose:
            print("\nSetting background on Linux OS.")

        current_env = os.environ.copy()
        current_env["DISPLAY"] = ":0" # in case run from where DISPLAY isn't set

        # Find out what window manager is currently in use.
        current_window_manager = current_env["XDG_CURRENT_DESKTOP"]
        if args.verbose:
            print("Desktop environment variable is XDG_CURRENT_DESKTOP =",
                  current_window_manager)

        if current_window_manager == "LXDE":
            # -w, --set-wallpaper=FILE   Set desktop wallpaper from image FILE
            # --wallpaper-mode=MODE      MODE=(color|stretch|fit|center|tile)
            try:
                subprocess.call(["pcmanfm", "--set-wallpaper", image_file_name,
                                 "--wallpaper-mode=fit"])
            except OSError as e:
                print("\nError attempting to run 'pcmanfm'.  Be sure the program"
                      "\nis installed on your system.  The image was apparently"
                      "\ncreated but there was an error in setting it as the"
                      "\nbackground image")

        elif current_window_manager == "X-Cinnamon":
            if args.verbose:
                print("Detected Cinnamon window manager, using Gnome calls.")
            current_window_manager = "GNOME"

        elif current_window_manager == "Unity":
            if args.verbose:
                print("Detected Unity window manager, using Gnome calls.")
            current_window_manager = "GNOME"

        elif current_window_manager != "GNOME":
            # Later more options may be added, but assume Gnome for now if not LXDE.
            if args.verbose:
                print("Assuming Gnome window manager and hoping for the best.")
            current_window_manager = "GNOME"

        if current_window_manager == "GNOME":
            # TODO we can detect Gnome 2 versus Gnome 3 by running
            #    gnome-session --version
            # and checking the output.  For Gnome 2 the commands should be
            #    gconftool-2 --type=string --set \
            #          /desktop/gnome/background/picture_filename <pathnameToImage>
            #    gconftool-2 --type=string --set \
            #          /desktop/gnome/background/picture_options stretched

            image_url = "file://" + image_file_name
            # Use gconf-editor to see the keys that can be set for a schema, go to
            #    / -> desktop -> gnome -> backgrounds
            # and click on a keys (it will list the values at the bottom).
            # There is also an option to gsettings to list the values, e.g.,
            #    gsettings list-keys org.gnome.desktop.background
            #    gsettings list-recursively org.gnome.desktop.background
            try:
                # os.system("GSETTINGS_BACKEND=dconf gsettings set "
                #          "org.gnome.desktop.background picture-options spanned")
                # os.system("GSETTINGS_BACKEND=dconf gsettings set "
                #          "org.gnome.desktop.background picture-uri \""+image_url+"\"")
                current_env["GSETTINGS_BACKEND"] = "dconf"
                subprocess.call(["gsettings", "set",
                                 "org.gnome.settings-daemon.plugins.background",
                                 "active", "true"], env=current_env)
                subprocess.call(["gsettings", "set", "org.gnome.desktop.background",
                                 "picture-options", "spanned"], env=current_env)
                subprocess.call(["gsettings", "set", "org.gnome.desktop.background",
                                 "picture-uri", image_url], env=current_env)
            except OSError as e:
                print("\nError attempting to run 'gsettings'.  Be sure the program is"
                      "\ninstalled on your system.  The image was apparently created"
                      "\nbut there was an error in setting it as the background."
                      "\nThe system reported:\n", e, file=sys.stderr)

    elif system_os == "Windows":
        if args.verbose:
            print("\nSetting background on Windows OS.")

        def set_wallpaper_mode():
            """Set the wallpaper mode to tiled.  Code modified from
            http://code.activestate.com/recipes/435877-change-the-wallpaper-under-windows/
            """
            if python_version[0] == "2":
                import _winreg
                winreg = _winreg
            else: import winreg

            wallpaper_style = '0'
            tile_wallpaper = '1' # use '0' for regular mode
            try:
                desktop_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                             'Control Panel\\Desktop',
                                             0,
                                             winreg.KEY_SET_VALUE)
                winreg.SetValueEx(desktop_key,
                                   'WallpaperStyle',
                                   0,
                                   winreg.REG_SZ,
                                   wallpaper_style)
                winreg.SetValueEx(desktop_key,
                                   'TileWallpaper',
                                   0,
                                   winreg.REG_SZ,
                                   tile_wallpaper)
                return True
            except:
                print("Warning: Exception encountered setting wallpaper mode",
                      file=sys.stderr)
                return False

        # import ctypes # now done at the top of the file
        SPI_SETDESKWALLPAPER = 0x14
        SPIF_UPDATEINIFILE = 0X01
        SPIF_SENDWININICHANGE = 0X02

        def set_background(image):
            """Set the background wallpaper to display."""
            try:
                if windows_release == "7":
                    ctypes.windll.user32.SystemParametersInfoW(
                        SPI_SETDESKWALLPAPER,
                        0,
                        image,
                        SPIF_UPDATEINIFILE | SPIF_SENDWININICHANGE)
                else: # XP, Vista, Windows 8
                    ctypes.windll.user32.SystemParametersInfoA(
                        SPI_SETDESKWALLPAPER,
                        0,
                        image,
                        SPIF_UPDATEINIFILE | SPIF_SENDWININICHANGE)
            except: # don't know what types of exceptions this might give
                print("\nError in makeSpanningBackground: Setting the background "
                      "wallpaper failed.", file=sys.stderr)

        set_wallpaper_mode() # set to tiled mode
        set_background(image_file_name)

    else:
        # TODO consider Cygwin implementation, but scipy is a pain to install
        # on Cygwin for now.
        print("\nSystem OS not recognized; not setting the generated image as"
              "\nthe current background wallpaper.", file=sys.stderr)


#
# The main code.
#


if __name__ == "__main__":

    # Parse the command-line arguments.
    args = parse_command_line_arguments(parser, help_string_replacement_pairs)

    # Set up the output file.
    save_file_name = process_path(args.outfile[0]) # make absolute, expand tilde
    extension = os.path.splitext(save_file_name)[1]
    if extension not in allowed_image_file_suffixes:
        path_error_exit(save_file_name, "No recognized image file suffix or"
                      " extension on the specified output filename.")
    dirname, filename = os.path.split(save_file_name)
    if not os.path.exists(dirname):
        path_error_exit(save_file_name, "The specified directory for the output"
                      " image file does not exist.")

    if os.path.exists(save_file_name) and not os.path.isfile(save_file_name):
        path_error_exit(save_file_name, "The specified output pathname exists but"
                      " is not a file.")

    # Handle --noclobber here, before possibly wasting time to create an image.
    if args.noclobber and os.path.exists(save_file_name):
        print("\nWarning from makeSpanningBackground: The specified output"
              " file\n   " + args.outfile[0] + "\nalready exists.  No file was"
              " written due to the noclobber option.")
        sys.exit(0)

    # Handle the --zoomspline option and its default value.
    zoom_spline = 3
    if args.zoomspline:
        zoom_spline = args.zoomspline[0]
        if zoom_spline > 5 or zoom_spline < 0:
            print("\nError in makeSpanningBackground: The specified spline order\n"
                  + str(zoom_spline) + " is not in the range 0-5.\n")
            sys.exit(1)

    # Print a welcome message if the verbose option was chosen.
    if args.verbose:
        if system_os == "Windows":
            print("\nRunning makeSpanningBackground on Windows..."
                  "\nWindows version:", windows_version)
        elif system_os == "Linux":
            print("\nRunning makeSpanningBackground on Linux..."
                  "\nLinux distribution:", linux_version)
        else:
            print("\nRunning makeSpanningBackground on an unknown OS...")

    # Now begin looping if the '--timedelay' option was set; if it was not
    # the loop will break after one execution.
    first_loop_completed = False
    while True:

        # Get the current display information.
        display_res_list = get_display_info()
        num_displays = len(display_res_list)
        if num_displays == 0:
            print("\nError in makeSpanningBackground: No displays detected."
                  "\nMaybe try explicitly setting the resolutions with the"
                  "\n'--reslist' option.\n")
            sys.exit(1)
        if args.verbose:
            print("\nDetected", num_displays, "displays:\n   ", display_res_list,
                  "\n")

        # Select an image for each display.
        bg_images = []
        bg_image_names = []
        # TODO use enumerate
        for disp_res, count in zip(display_res_list, range(len(display_res_list))):
            image = get_next_background_image(disp_res, display_res_list)
            if not image:
                print("\nError in makeSpanningBackground: No suitable image files"
                      "\nfound for display", str(count)+".\n", file=sys.stderr)
                sys.exit(1)
            if args.verbose:
                print("Image selected for display", count, "is\n   ", image[0], "\n")
            bg_image_names.append(image[0])
            bg_images.append(image[1])
            if args.oneimage:
                break

        if args.logcurrent:
            expanded_log_name = process_path(args.logcurrent[0]) # tilde expand
            # TODO make sure this is not a directory first
            current_images_log = open(expanded_log_name, "w")
            for count, img in enumerate(bg_image_names):
                print("Image on display", count, "is\n   ", img, "\n",
                      file=current_images_log)
            current_images_log.close()

        # Make the large, combined image.
        giant_image = create_giant_image(bg_images, display_res_list)

        # Write out the giant image to a file.
        try:
            if args.verbose:
                print("\nWriting the combined image to the file\n   "+save_file_name)
            sp.misc.imsave(save_file_name, giant_image)
        except IOError as e:
            print("\nWarning from makeSpanningBackground: Could not save to file"
                  "\n   " + save_file_name, "\nThe reported error was:\n", e,
                  file=sys.stderr)

        if not args.dontapply:
            if args.verbose:
                print("\nSetting the new image as the current background wallpaper.")
                if system_os == "Windows":
                    print("(Be sure to set wallpaper mode to 'tiled' in Windows.)")
            set_image_as_current_wallpaper(save_file_name)

        first_loop_completed = True
        if not args.timedelay:
            break
        if args.verbose:
            print("\n"+"-"*10, "Sleeping for", args.timedelay[0], "minutes.", "-"*30)
        time.sleep(args.timedelay[0]*60.0)

    if args.verbose:
        print("\nFinished execution of makeSpanningBackground.")

