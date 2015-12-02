cd %~dp0
git commit -m "before newly ignored remove"
git rm -r --cached .
git add .
git commit -m ".gitignore is now working"
