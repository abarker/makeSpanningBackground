# makeSpanningBackground

Selects and sets the background wallpaper on multiple displays/monitors (and
single monitors also).

The makeSpanningBackground program runs on both Linux and Windows versions of
Python.  Most of the testing has been on one and two monitor systems with
Python 2.7 or 3.X, on Ubuntu and Windows 7, but Fedora, Vista, and Windows XP
have also had some testing.  Note that under KDE and Xfce on Linux the
generated image may not be automatically set to be the wallpaper (with the
"spanned" mode auto-set); you'll have to apply those settings by hand.

As of Oct 2014 this program works on Cinnamon and Unity, at least, but Gnome
and LXDE have been changing the way they handle background images so it may not
work.  There is now a Bash script in the bin directory which can be used as a
simplified frontend to the full program on Linux systems.

## Dependencies

The dependencies are as follows:
```
   Linux with Python 2.x (at least Python 2.6):
                   Ubuntu                               Fedora
      numpy       sudo apt-get install python-numpy    yum install numpy
      scipy       sudo apt-get install python-scipy    yum install scipy
      PIL         sudo apt-get install python-imaging  yum install python-imaging
      xrandr      pre-installed for most distributions 
      gsettings   pre-installed for most distributions

   Linux with Python 3.x:
      Same as for Python 2.x.  For numpy and scipy just apt-get the version
      that starts with python3, e.g., python3-numpy.

   Windows with Python 2.x or 3.x (at least Python 2.6):
      Needs numpy, scipy, and PIL.  Windows users who want binaries and are not
      using a package like Anaconda or Python(x,y) where the dependencies are
      pre-installed might try the following (get the version to match the
      installed Python):

      http://www.lfd.uci.edu/~gohlke/pythonlibs/#pil
      http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
      http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy

      Note that on Windows the background mode must be set to "tiled" in the
      standard wallpaper-selection dialog window.  The program will attempt
      to do this, but if that doesn't work it will need to be set by the user.
      On XP you may need to always choose the .bmp format for the saved image
      files.

   Windows Cygwin Python
      The dependencies are the same as in Windows.  Cygwin comes with 
      packages for python-argparse, python-imaging, and python-numpy.
      Unfortunately, scipy is apparently difficult to install in Cygwin.
      You can run the script with Windows Python from a Cygwin terminal.
```
## Documentation

For the general documentation, run
```
python makeSpanningBackground.py --help
```
The output of that command follows.
```
Usage: makeSpanningBackground.py [-h] -o OUTFILE_NAME [-v] [-1]
                                 [-f R_VAL G_VAL B_VAL] [-t MINUTES]
                                 [-p PCT_FLOAT] [-z ONE_OF_012345] [-s]
                                 [-c R_VAL G_VAL B_VAL] [-R] [-d]
                                 [--noclobber] [-r RESLIST [RESLIST ...]]
                                 [-w X_POS Y_POS] [-x] [-L FNAME]
                                 IMAGE_FILE_OR_DIR [IMAGE_FILE_OR_DIR ...]

Description:

     Make a single, combined background image file from separate background
     image files, one for each display. The default is to get information
     from the system about the display resolutions, and to randomly choose
     an image for each display from the image files and image directories
     provided. Sampling is without replacement; the full list is regenerated
     when it is empty (including any changes which might have occurred in
     the directories).

     The positional arguments are image files and directories containing
     image files. The required flag '--outfile' or '-o' specifies the
     pathname of the file where the combined image file will be written (or
     silently overwritten). An example:

        python makeSpanningBackground.py ~/bgDir photo.jpg -o combo.bmp

     On Linux the program file can be made executable and the initial
     "python" above can be omitted if the Python version at
     "/usr/bin/python" is the correct one (if not, the first line in the
     program can be modified). Windows users can achieve the same effect by
     renaming the script to have a '.py' suffix and making sure '.PY' is in
     the PATHEXT list. On any system the program file (or a script calling
     it) can be placed somewhere in the PATH to avoid having to supply the
     path.

     To use this kind of combined background image in Linux the background
     mode should be set to 'spanned'; the program will attempt to set this
     automatically. On Windows the background mode should be 'tiled', but
     the mode must be explicitly set by the user (usually from the system's
     background-setting window). In Windows XP the '.bmp' format should be
     chosen for the output file (simply by using that suffix on the output
     filename). Also, backgrounds set in Windows XP do not persist between
     logins and must be reset.

     Note that this program can be set as a startup program to change the
     background wallpaper on logins. When the '-t <minutes>' command switch
     is used the program will infinitely loop, waiting for the specified
     number of minutes between iterations.

     It is convenient to create a simple shell-script wrapper or batch file
     to call the program with the "usual" command-line arguments.

Positional arguments:

  IMAGE_FILE_OR_DIR     A whitespace-separated list of the pathnames of image
                        files and/or directories containing image files. Use
                        quotes around any file or directory name which
                        contains a space. Pathnames can be repeated on the
                        list, and the list will be reloaded if it becomes
                        empty. A specified filename will be silently ignored
                        if it does not have a suffix in the list ['.BMP',
                        '.DCX', '.DIB', '.EPS', '.GIF', '.IM', '.JPE',
                        '.JPEG', '.JPG', '.PBM', '.PCD', '.PCX', '.PDF',
                        '.PGM', '.PNG', '.PPM', '.PS', '.PSD', '.TIF',
                        '.TIFF', '.XBM', '.XPM', '.bmp', '.dcx', '.dib',
                        '.eps', '.gif', '.im', '.jpe', '.jpeg', '.jpg',
                        '.pbm', '.pcd', '.pcx', '.pdf', '.pgm', '.png',
                        '.ppm', '.ps', '.psd', '.tif', '.tiff', '.xbm',
                        '.xpm'].


Optional arguments:

  -h, --help            Show this help message and exit.

  -o OUTFILE_NAME, --outfile OUTFILE_NAME
                        The pathname of the output file. This flag is
                        required. Any existing file by that name will be
                        silently overwritten. The filename can have any suffix
                        that ndimage (via PIL) can handle, and the output file
                        will be written in that format. Not all formats will
                        necessarily be settable as background images. Common
                        choices are '.jpg' and '.bmp', where the latter
                        produces files which are larger but of higher quality.
                        Note that Windows XP requires the '.bmp' format.

  -v, --verbose         Print more information about the program's actions and
                        progress. Without this switch only error messages are
                        printed to the screen.

  -1, --oneimage        Use a single background image, scaled to stretch over
                        all the displays. This is currently based on the
                        screen resolutions, not the physical sizes of the
                        monitors (dpi).

  -f R_VAL G_VAL B_VAL, --fitimage R_VAL G_VAL B_VAL
                        If this option is set then images are scaled to fully
                        fit into the corresponding display. The three
                        arguments are RGB byte values specifying the color of
                        any region not covered by an image. Use '-f 0 0 0' for
                        black. The default program behavior without this
                        switch is to use the minimum scaling to completely
                        fill the display while preserving the aspect ratio.

  -t MINUTES, --timedelay MINUTES
                        If this option is set the program will infinitely
                        loop. On each iteration it will collect the current
                        display information, re-select images for each
                        display, create a combined image, and set it as the
                        background (assuming that behavior is not modified by
                        any other options). The floating point argument gives
                        the number of minutes to sleep between iterations. The
                        image list will only be reloaded when it becomes
                        empty, so every image will be used exactly once before
                        any images are reloaded.

  -p PCT_FLOAT, --percenterror PCT_FLOAT
                        The percentage of an image's area which is allowed to
                        be cropped-out when scaling it to fit a display
                        resolution. In '--fitimage' mode it is the percent of
                        non-image in the display area. In '--oneimage' mode it
                        is the percent of the image that does not fall onto
                        any display. Set this option to zero for exact fit
                        only. The program will exit with an error message if
                        it cannot find enough suitable images.

  -z ONE_OF_012345, --zoomspline ONE_OF_012345
                        Set the order of the spline used by ndimage to resize
                        (zoom) images. The value can be from 0 to 5, with 3
                        the default. Lower orders are faster, higher orders
                        have better quality. (Also, for higher-quality images:
                        output files in '.bmp' format are larger but tend to
                        look better.)

  -s, --sequential      Process images sequentially as listed in the
                        positional arguments, with the files in any image
                        directories forming alphabetical sublists. This can be
                        used to force specific images to appear on specific
                        displays. On Linux the ordering of the displays is the
                        same as in xrandr. If you are unsure about the
                        ordering, try some images and see where they end up.
                        If display resolutions and offsets are explicitly set
                        with '--reslist' and '--sequential' is set then the
                        image files will corresponding one to one with the
                        list of resolution specifiers.

  -c R_VAL G_VAL B_VAL, --colorfill R_VAL G_VAL B_VAL
                        The three arguments are RGB byte values specifying the
                        color of any region in the large, combined image (a
                        bounding-box on the displays) which is not covered by
                        a display. Use '-c 0 0 0' for black.

  -R, --recursive       Recursively search any supplied image directories for
                        image files.

  -d, --dontapply       Do not attempt to apply the created image as the
                        working background image. The program will simply exit
                        after writing the image file.

  --noclobber           Never overwrite an existing file as the output
                        file.

  -r RESLIST [RESLIST ...], --reslist RESLIST [RESLIST ...]
                        Set the resolutions and offsets to use. No system
                        lookup will be attempted. This must be a space-
                        separated list of strings of the form
                        x*y+xOffset+yOffset (like in the output of xrandr on
                        Linux), where xOffset and yOffset give the top-left
                        position of the display and x and y are the display
                        sizes (resolution). Like in X11, (0,0) is assumed to
                        be at the top left of a bounding box on all the
                        displays. For a Windows machine the top left of the
                        primary display will also need to be defined, using
                        the '--windows' option described below. This option
                        cannot be immediately followed by the positional
                        arguments.

  -w X_POS Y_POS, --windows X_POS Y_POS
                        This option should only be necessary on a Windows
                        machine when using the '--reslist' option. The two
                        arguments x and y (in that order) should be set to the
                        top left position of the primary display. This is
                        necessary because, unlike X11, Windows assumes (0,0)
                        is at the top left of the primary display. Other
                        displays can then have negative locations. This
                        program translates Windows positions so that all
                        addresses are positive, with (0,0) at the top left of
                        all the displays (which is how positions should be
                        entered for the '--reslist' option). The program then
                        corrects for this translation in a final step, which
                        wraps the image into tiled mode relative to the
                        primary display. This requires knowing the top left
                        position of the primary display.

  -x, --x11             Create an X11 type image even if Windows is detected
                        as the OS. When this option is selected no final
                        "modular wrap" will be applied to correct for the
                        position of the primary display.

  -L FNAME, --logcurrent FNAME
                        Write the names of the current images to a file. The
                        single argument is the name of the file to write the
                        filenames to. Useful when you want to know the
                        filenames of the images being displayed.

The makeSpanningBackground program is Copyright (c) 2012 by Allen Barker.
Released under the MIT license.
```
