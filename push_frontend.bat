@echo off
cd /d "C:\Users\rnuser\Documents\deepresearch\frontend"
echo Checking git status...
git status
echo.
echo Adding Netlify configuration files...
git add netlify.toml
git add public/_headers
git add public/_redirects
git add src/config/api.ts
echo.
echo Committing changes...
git commit -m "Fix Netlify deployment: Add netlify.toml, update API config, add headers and redirects"
echo.
echo Pushing to repository...
git push origin main
echo.
echo Done!
pause
