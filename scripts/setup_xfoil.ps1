$targetPath = "C:\Users\abhis\XFOIL6.99"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

if ($currentPath -notlike "*$targetPath*") {
    $newPath = $currentPath + ";" + $targetPath
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Output "Successfully added $targetPath to User PATH."
    Write-Output "Please restart your terminal for changes to take effect."
} else {
    Write-Output "Path already exists in User PATH."
}

# Verify locally in this session for confirmation
$env:Path += ";$targetPath"
if (Get-Command xfoil -ErrorAction SilentlyContinue) {
    Write-Output "Verification: 'xfoil' command found!"
} else {
    Write-Output "Verification: 'xfoil' command NOT found."
}
