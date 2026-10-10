@echo off
:: Run once: puts a "Frame Selector" shortcut on the desktop that launches start_frame_selector.bat.
set PROJECT_DIR=%~dp0
set TARGET=%PROJECT_DIR%start_frame_selector.bat
set ICON=%PROJECT_DIR%frontend\src-tauri\icons\icon.ico

powershell -NoProfile -Command ^
  "$d=[Environment]::GetFolderPath('Desktop');" ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut(\"$d\Frame Selector.lnk\");" ^
  "$s.TargetPath='%TARGET%';" ^
  "$s.WorkingDirectory='%PROJECT_DIR%';" ^
  "$s.Description='Pick the frames worth labelling from traffic videos';" ^
  "if (Test-Path '%ICON%') { $s.IconLocation='%ICON%' };" ^
  "$s.Save();" ^
  "Write-Host ('Shortcut created: ' + $d + '\Frame Selector.lnk')"

if errorlevel 1 (
    echo ERROR: could not create the shortcut.
    echo You can also right-click start_frame_selector.bat, then Send to ^> Desktop ^(create shortcut^).
)
pause
