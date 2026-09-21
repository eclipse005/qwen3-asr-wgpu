# Interleaved A/B for one env switch.
#
# Sequential comparisons are worthless on this box: the same configuration
# measures 22.3 and 23.3 RTFx half an hour apart, so a "before then after" run
# invents ~4% of effect.  This runs both arms once per round and reports medians,
# with the transcript verdict as a gate.
#
#   pwsh -File scripts/ab.ps1 -Var QASR_GQA_CHUNK -A 256 -B 512 -Wav 90s_en -Reps 3
param(
    [Parameter(Mandatory = $true)][string]$Var,
    [Parameter(Mandatory = $true)][string]$A,
    [Parameter(Mandatory = $true)][string]$B,
    [Parameter(Mandatory = $true)][string]$Wav,
    [int]$Reps = 3,
    [string]$Size = '0.6B',
    [int]$MaxNew = 1024,
    # `transcribe` prints several phases; compare whichever one the change is
    # supposed to move so a win in one phase can't hide a loss in another.
    [ValidateSet('rtfx', 'mel', 'enc', 'pre', 'dec')][string]$On = 'rtfx'
)
$ErrorActionPreference = 'Stop'
# Without this a CJK transcript decodes as the console code page and swallows
# the newline that separates it from the MATCH verdict.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repo = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $repo 'target\release\transcribe.exe'
$model = "D:\Qwen3-ASR\models\Qwen3-ASR-$Size-hf"
$wavPath = "D:\Qwen3-ASR\fixtures\$Wav.wav"
$base = "D:\qwen3-asr-rs\docs\baseline\texts\python-hf_${Size}_$Wav.txt"

$acc = @{ $A = @(); $B = @() }
$verdict = @{ $A = 'MATCH'; $B = 'MATCH' }
foreach ($rep in 1..$Reps) {
    foreach ($arm in @($A, $B)) {
        Set-Item -Path "Env:$Var" -Value $arm
        $lines = & $exe --model $model --wav $wavPath --max-new $MaxNew --baseline $base 2>&1 |
            ForEach-Object { "$_" }
        $text = $lines -join "`n"
        $ph = [regex]::Match($text, 'mel=(\d+)ms enc=(\d+)ms prefill=(\d+)ms decode=(\d+)ms')
        $vs = @($lines | Where-Object { "$_" -like 'vs *: *' }) | Select-Object -Last 1
        if (-not ("$vs" -like '*: MATCH')) { $verdict[$arm] = 'MISMATCH' }
        $acc[$arm] += [pscustomobject]@{
            rtfx = [double][regex]::Match($text, 'RTFx=([\d.]+)').Groups[1].Value
            mel  = [int]$ph.Groups[1].Value
            enc  = [int]$ph.Groups[2].Value
            pre  = [int]$ph.Groups[3].Value
            dec  = [int]$ph.Groups[4].Value
        }
    }
}

$median = {
    param($rows, $field)
    $sorted = @($rows | Sort-Object $field)
    $sorted[[int]($sorted.Count / 2)]
}

"$Var  A=$A  B=$B   fixture $Size/$Wav   reps=$Reps   compare on '$On'"
"{0,-8} {1,-28} {2,-10} {3}" -f 'arm', 'per rep', 'median', 'verdict'
$med = @{}
foreach ($arm in @($A, $B)) {
    $m = & $median $acc[$arm] $On
    $med[$arm] = $m
    "{0,-8} {1,-28} {2,-10:F3} {3}" -f $arm,
        (($acc[$arm] | ForEach-Object { "{0:F2}" -f $_.$On }) -join ' '),
        $m.$On, $verdict[$arm]
}
$va = $med[$A].$On
$vb = $med[$B].$On
"delta: {0} {1:+0.0%;-0.0%}   (decode {2:+0;-0} ms, prefill {3:+0;-0} ms)" -f `
    $On, (($vb - $va) / $va), ($med[$B].dec - $med[$A].dec), ($med[$B].pre - $med[$A].pre)
if ($verdict[$A] -eq 'MISMATCH' -or $verdict[$B] -eq 'MISMATCH') { exit 1 }
