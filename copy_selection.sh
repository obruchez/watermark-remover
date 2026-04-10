#!/bin/bash

SRC="/Volumes/images/Appareil photo/2026/2026.03.29 - Course des Ânes [Kuva, avec watermark]"
DST="/Users/olivierbruchez/Github/watermark-remover/photos"
REF="/Volumes/images/Appareil photo/2026/2026.03.29 - Course des Ânes (sélection Adriel, Aline et Olivier) [Kuva, sans watermark, v1]"

mkdir -p "$DST"

copied=0
missing=0

for ref_file in "$REF"/*; do
    # Strip .v1 from filename to get the source filename
    basename=$(basename "$ref_file")
    src_name="${basename/.v1.webp/.webp}"

    if [[ -f "$SRC/$src_name" ]]; then
        cp "$SRC/$src_name" "$DST/$src_name"
        echo "Copied: $src_name"
        ((copied++))
    else
        echo "NOT FOUND: $src_name (from $basename)"
        ((missing++))
    fi
done

echo ""
echo "Done. Copied: $copied, Missing: $missing"