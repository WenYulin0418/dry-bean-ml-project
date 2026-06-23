param(
    [string]$DocumentPath = ""
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

if (-not $DocumentPath) {
    $DocumentPath = (
        Get-ChildItem -LiteralPath "paper" -Filter "*.docx" -File |
        Select-Object -First 1
    ).FullName
}
$document = (Resolve-Path -LiteralPath $DocumentPath).Path
$work = Join-Path (Split-Path $document -Parent) ".paper_patch"
$sourceZip = Join-Path $work "source.zip"
$expanded = Join-Path $work "expanded"
$rebuilt = Join-Path $work "rebuilt.docx"
$backup = "$document.bak"

if (Test-Path -LiteralPath $work) {
    Remove-Item -LiteralPath $work -Recurse -Force
}
New-Item -ItemType Directory -Path $work | Out-Null
Copy-Item -LiteralPath $document -Destination $sourceZip
[System.IO.Compression.ZipFile]::ExtractToDirectory($sourceZip, $expanded)

$documentXmlPath = Join-Path $expanded "word\document.xml"
[xml]$documentXml = Get-Content -LiteralPath $documentXmlPath -Raw -Encoding UTF8
$namespace = New-Object System.Xml.XmlNamespaceManager($documentXml.NameTable)
$namespace.AddNamespace(
    "w",
    "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
)

function Set-RowValue {
    param(
        [xml]$Xml,
        [System.Xml.XmlNamespaceManager]$Namespace,
        [string]$Label,
        [string]$Value
    )
    $rows = $Xml.SelectNodes("//w:tr", $Namespace)
    foreach ($row in $rows) {
        $cells = $row.SelectNodes("./w:tc", $Namespace)
        if ($cells.Count -lt 2) {
            continue
        }
        $labelText = ($cells[0].SelectNodes(".//w:t", $Namespace) |
            ForEach-Object { $_.InnerText }) -join ""
        if ($labelText -eq $Label) {
            $texts = $cells[1].SelectNodes(".//w:t", $Namespace)
            if ($texts.Count -eq 0) {
                throw "Cover field '$Label' has no editable text node"
            }
            $texts[0].InnerText = $Value
            for ($index = 1; $index -lt $texts.Count; $index++) {
                $texts[$index].InnerText = ""
            }
            return
        }
    }
    throw "Cover field '$Label' was not found"
}

$nameLabel = ([char[]](0x59D3, 0x540D)) -join ""
$studentIdLabel = ([char[]](0x5B66, 0x53F7)) -join ""
$classLabel = ([char[]](0x73ED, 0x7EA7)) -join ""
$studentName = ([char[]](0x6587, 0x6E1D, 0x9716)) -join ""

Set-RowValue -Xml $documentXml -Namespace $namespace `
    -Label $nameLabel -Value $studentName
Set-RowValue -Xml $documentXml -Namespace $namespace `
    -Label $studentIdLabel -Value "202493030"

$classRows = @($documentXml.SelectNodes("//w:tr", $namespace) | Where-Object {
    $rowText = (($_.SelectNodes(".//w:t", $namespace) |
        ForEach-Object { $_.InnerText }) -join "")
    $rowText.StartsWith($classLabel)
})
foreach ($row in $classRows) {
    [void]$row.ParentNode.RemoveChild($row)
}

$documentXml.Save($documentXmlPath)

$wordXmlFiles = Get-ChildItem -LiteralPath (Join-Path $expanded "word") `
    -Filter "*.xml" -File -Recurse
foreach ($file in $wordXmlFiles) {
    $content = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
    $content = [regex]::Replace(
        $content,
        '(<w:color\b[^>]*\bw:val=")(?!auto)[^"]+(")',
        '${1}000000${2}'
    )
    Set-Content -LiteralPath $file.FullName -Value $content `
        -Encoding UTF8 -NoNewline
}

$corePath = Join-Path $expanded "docProps\core.xml"
if (Test-Path -LiteralPath $corePath) {
    [xml]$core = Get-Content -LiteralPath $corePath -Raw -Encoding UTF8
    $creator = $core.SelectSingleNode(
        "//*[local-name()='creator']"
    )
    if ($creator) {
        $creator.InnerText = $studentName
    }
    $core.Save($corePath)
}

if (Test-Path -LiteralPath $rebuilt) {
    Remove-Item -LiteralPath $rebuilt -Force
}
$outputStream = [System.IO.File]::Open(
    $rebuilt,
    [System.IO.FileMode]::CreateNew
)
$archive = New-Object System.IO.Compression.ZipArchive(
    $outputStream,
    [System.IO.Compression.ZipArchiveMode]::Create
)
try {
    foreach ($file in Get-ChildItem -LiteralPath $expanded -File -Recurse) {
        $relative = $file.FullName.Substring($expanded.Length + 1)
        $entryName = $relative.Replace("\", "/")
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $archive,
            $file.FullName,
            $entryName,
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
}
finally {
    $archive.Dispose()
    $outputStream.Dispose()
}

if (-not (Test-Path -LiteralPath $backup)) {
    Copy-Item -LiteralPath $document -Destination $backup
}
$finalDocument = $document
try {
    Copy-Item -LiteralPath $rebuilt -Destination $document -Force
}
catch [System.IO.IOException] {
    $directory = Split-Path $document -Parent
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($document)
    $finalDocument = Join-Path $directory ($stem + "_revised.docx")
    Copy-Item -LiteralPath $rebuilt -Destination $finalDocument -Force
}

Remove-Item -LiteralPath $work -Recurse -Force
Write-Output "Patched: $finalDocument"
