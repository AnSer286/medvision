@echo off
echo ================================
echo PUSH changes to repository
echo ================================
cd /d "%~dp0"

REM Getting current branch
for /f %%i in ('git branch --show-current') do set current_branch=%%i
echo Current branch: %current_branch%

REM Showing status
echo.
echo Changes status:
git status --short

echo.
set /p commit_msg="Enter commit message (or press Enter for 'Update'): "
if "%commit_msg%"=="" set commit_msg=Update

echo.
echo Attaching updates...
git add .

echo Committing with the message: "%commit_msg%"
git commit -m "%commit_msg%"

echo Pushing to origin/%current_branch%...
git push origin %current_branch%

echo.
echo Done!
pause