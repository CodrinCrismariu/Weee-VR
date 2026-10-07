#Requires -RunAsAdministrator
# Passes the USB-connected iPhone through to WSL (Ubuntu-24.04) with usbipd-win.
# Run from an *admin* PowerShell after plugging in the iPhone, unlocking it and tapping "Trust".
# Use -Force if attach reports the device is busy (Windows then can't use the iPhone until you run
# `usbipd unbind --busid <id>`).
param([switch]$Force)

$line = usbipd list | Select-String '05ac:' | Select-Object -First 1
if (-not $line) {
    Write-Error "No Apple device found. Plug in the iPhone, unlock it and tap Trust."
    exit 1
}
$busid = ($line.ToString().Trim() -split '\s+')[0]
Write-Host "iPhone found at bus id $busid"

wsl -d Ubuntu-24.04 -- true   # make sure the distro is running

if ($Force) { usbipd bind --busid $busid --force } else { usbipd bind --busid $busid }
# --auto-attach keeps re-attaching when the iPhone re-enumerates (pairing, Developer Mode reboot).
# Leave this window open while developing.
usbipd attach --wsl --busid $busid --auto-attach
