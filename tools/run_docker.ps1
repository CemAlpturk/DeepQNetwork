# Powershell script for running docker container.

# Script assumes that the image is named "frtn70_env".
$image="frtn70_env"

# Get full path to root directory.
$RootDirectory = (get-item $PSScriptRoot).parent.FullName
$srcVolume = $RootDirectory + ":/repo"
Write-Output "Volume: $srcVolume"

docker run -it -v $srcVolume $image bash
