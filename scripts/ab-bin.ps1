# Interleaved A/B for a change with *no runtime knob* -- two binaries.
#
# `ab.ps1` toggles an env var, so it cannot test anything baked into the shaders
# (a layout, a workgroup shape, a tile geometry).  Sequential before/after runs
# are worthless here for the same reason they are worthless there: the same
# configuration measures 22.3 and 23.3 RTFx half an hour apart, so a sequential
# comparison invents ~4% of effect.
#
#   pwsh -File scripts/ab-bin.ps1 -A transcribe_base.exe -B transcribe_new.exe \
#        -Wav 180s_zh -Reps 3 -On dec
#
# Both binaries must be in target\release.  The transcript verdict of each arm is
# printed; a MISMATCH arm means its numbers are measuring a different token
# stream and the comparison is void.
param(
    [Parameter(Mandatory = $true)][string]$A,
    [Parameter(Mandatory = $true)][string]$B,
    [Parameter(Mandatory = $true)][string]$Wav,
    [int]$Reps = 3,
    [string]$Size = '0.6B',
    [int]$MaxNew = 1024,
    [ValidateSet('rtfx', 'mel', 'enc', 'pre', 'dec')][string]$On = 'dec'
)
$ErrorActionPreference = 'Stop'
# Without this a CJK transcript decodes as the console code page and swallows the
# newline that separates it from the MATCH verdict.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repo = Split-Path $PSScriptRoot -Parent
$exes = @{ $A = Join-Path $repo "target\release\$A"; $B = Join-Path $repo "target\release\$B" }
foreach ($arm in @($A, $B)) {
    if (-not (Test-Path $exes[$arm])) { throw "missing binary $($exes[$arm])" }
}
$model = "D:\Qwen3-ASR\models\Qwen3-ASR-$Size-hf"
$wavPath = "D:\Qwen3-ASR\fixtures\$Wav.wav"
$base = "D:\qwen3-asr-rs\docs\baseline\texts\python-hf_${Size}_$Wav.txt"

$acc = @{ $A = @(); $B = @() }
$verdict = @{ $A = 'MATCH'; $B = 'MATCH' }
foreach ($rep in 1..$Reps) {
    foreach ($arm in @($A, $B)) {
        $lines = & $exes[$arm] --model $model --wav $wavPath --max-new $MaxNew --baseline $base 2>&1 |
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

"A=$A  B=$B   fixture $Size/$Wav   reps=$Reps   compare on '$On'"
"{0,-24} {1,-28} {2,-10} {3}" -f 'arm', 'per rep', 'median', 'verdict'
$med = @{}
foreach ($arm in @($A, $B)) {
    $m = & $median $acc[$arm] $On
    $med[$arm] = $m
    "{0,-24} {1,-28} {2,-10:F3} {3}" -f $arm,
        (($acc[$arm] | ForEach-Object { "{0:F2}" -f $_.$On }) -join ' '),
        $m.$On, $verdict[$arm]
}
$va = $med[$A].$On
$vb = $med[$B].$On
"delta: {0} {1:+0.0%;-0.0%}   (decode {2:+0;-0} ms, prefill {3:+0;-0} ms)" -f `
    $On, (($vb - $va) / $va), ($med[$B].dec - $med[$A].dec), ($med[$B].pre - $med[$A].pre)
if ($verdict[$A] -eq 'MISMATCH' -or $verdict[$B] -eq 'MISMATCH') { exit 1 }
