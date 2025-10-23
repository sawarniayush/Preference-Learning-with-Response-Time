#!/bin/bash
# Move all contents from data_files_backup/ to the parent directory

main_folder="."
src_folder="$main_folder/data_files_backup"

# safety check
if [ ! -d "$src_folder" ]; then
  echo "❌ Source folder not found: $src_folder"
  exit 1
fi

# move everything (files + subfolders)
echo "📦 Moving contents from $src_folder to $main_folder ..."
shopt -s dotglob  # include hidden files
for item in "$src_folder"/*; do
  if [ -e "$item" ]; then
    base=$(basename "$item")
    dest="$main_folder/$base"

    # handle name conflicts
    if [ -e "$dest" ]; then
      echo "⚠️ Conflict: $dest exists. Renaming to ${base}_from_backup"
      dest="${main_folder}/${base}_from_backup"
    fi

    mv "$item" "$dest"
  fi
done
shopt -u dotglob

echo "✅ Done! All contents moved to $main_folder"
