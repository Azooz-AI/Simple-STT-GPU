Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "E:\Automations\STT-Windows-Tool"
WshShell.Run "venv\Scripts\pythonw.exe main.py", 0
Set WshShell = Nothing