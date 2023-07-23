# Running jlweather versions on Crusher

"crusher_commands.sh" is a shell script that runs jlweather using Makefile.

After changing "ACCONT" and "USERNAME" with your info on Crusher, you can run a version of jlweather by uncommenting each command.

The compiler used in this script is a Cray Fortran compile and AMD hip compiler. replace them if you want to use another compiler.

In case you are trying to run them, copy the files in this folder to other folder and modify compiler command and "WORK_DIR" too.
