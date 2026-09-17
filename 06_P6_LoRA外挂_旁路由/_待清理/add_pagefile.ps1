$ErrorActionPreference = "Stop"
try {
    $pf = Set-WmiInstance -Class Win32_PageFileSetting -Arguments @{Name="H:\pagefile.sys"; InitialSize=16384; MaximumSize=32768}
    "OK " + $pf.Name | Out-File C:\temp_pf.txt -Encoding utf8
} catch {
    $_.Exception.Message | Out-File C:\temp_pf.txt -Encoding utf8
}
