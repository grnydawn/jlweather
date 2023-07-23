# Running jlweather versions on Crusher

crusher_commands.sh is a shell script that runs jlweather using Makefile.

After changing the values of ACCOUNT and USERNAME to your information on Crusher, you can run a version of jlweather by uncommenting each command.

The compiler used in this script is a Cray Fortran compiler and an AMD HIP compiler. Replace them if you want to use another compiler.

In case you are trying to run them on another machine, copy the files in this folder to another folder and modify the compiler command and WORK_DIR as well.
