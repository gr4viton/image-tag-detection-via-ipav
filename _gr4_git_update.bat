D:
cd D:\DEV\PYTHON\pyCV\2015_10_11 - pycmulti\

echo Current working directory is %cd%
echo Changing direcotory to %cd%

git.exe add .
git.exe commit -m "%date% %time%"
git.exe pull -v --no-rebase --progress "origin"
git.exe commit -m "%date% %time% after pull"
git.exe push origin master

python.exe -c "print('\7')" :: BEEP

set /a time2wait=%1	:: from first parameter
echo %time2wait%
if %time2wait% NEQ 0 (
	timeout %time2wait%
) ELSE (
	timeout 3
)
