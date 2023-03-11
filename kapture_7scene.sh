cd $1

find . -iname "*.zip" -print0 | xargs -n1 -0 -I {} unzip {}

kapture_import_7scenes.py -i . -o ./kapture/mapping -p mapping
kapture_import_7scenes.py -i . -o ./kapture/query -p query