@echo off
cd /d "C:\Users\rnuser\Documents\deepresearch\backend"
echo Checking git status...
git status
echo.
echo Adding all changes...
git add .
echo.
echo Committing changes...
git commit -m "Backend updates and restructuring"
echo.
echo Pushing to repository...
git push origin main
echo.
echo Done!
pause
