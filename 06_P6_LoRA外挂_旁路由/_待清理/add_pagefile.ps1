$ErrorActionPreference = "Stop"
try {
    $pf = Set-WmiInstance -Class Win32_PageFileSetting -Arguments @{Name="<path>\pagefile.sys"; InitialSize=16384; MaximumSize=32768}
    "OK " + $pf.Name | Out-File <path>\temp_pf.txt -Encoding utf8
} catch {
    $_.Exception.Message | Out-File <path>\temp_pf.txt -Encoding utf8
}
