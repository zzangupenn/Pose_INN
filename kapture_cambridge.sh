cd $1

sed 's/.jpg/.png/g' ./reconstruction.nvm > ./reconstruction_png.nvm
tail -n +4 ./dataset_train.txt > ./dataset_train_cut.txt
cut -d\  -f1 ./dataset_train_cut.txt > ./dataset_train_list.txt
tail -n +4 ./dataset_test.txt > ./dataset_test_cut.txt
cut -d\  -f1 ./dataset_test_cut.txt > ./dataset_test_list.txt

kapture_import_nvm.py -v info -i ./reconstruction_png.nvm -im ./ -o ./kapture/mapping --filter-list ./dataset_train_list.txt

kapture_import_nvm.py -v info -i ./reconstruction_png.nvm -im ./ -o ./kapture/query --filter-list ./dataset_test_list.txt
