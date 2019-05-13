Get-ChildItem -Directory | ForEach-Object {
    Copy-Item $_.FullName -Destination ..\ -force -Recurse
}

"Minimods Copied Successfully."
pause