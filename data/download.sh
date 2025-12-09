curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip

echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -
echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip"    | sha1sum -c -

unzip fma_metadata.zip -d data/
unzip fma_small.zip -d data/

rm -f data/fma_small/099/099134.mp3
rm -f data/fma_small/108/108925.mp3
rm -f data/fma_small/133/133297.mp3


rm -f fma_metadata.zip
rm -f fma_small.zip